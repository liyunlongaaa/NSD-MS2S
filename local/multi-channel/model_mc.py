# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from resnet import ResNet101
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from WavLM import WavLM, WavLMConfig
from conformer import ConformerBlock


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


def make_attention_layer(d_model, nhead, dropout=0.1, num_layers=1):
    model = nn.Sequential()
    for l in range(num_layers):
        model.add_module(f'att_{l+1}', Attention(d_model, nhead, dropout=dropout))
    return model

def make_conformer_block(conf):
    #dim = 256, dim_head = 64, heads = 4, ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = 31, attn_dropout = 0., ff_dropout = 0., conv_dropout = 0.
    conformer_block = ConformerBlock( 
                                dim = conf["dim"],
                                dim_head = conf["dim_head"],
                                heads = conf["heads"],
                                ff_mult = conf["ff_mult"],
                                conv_expansion_factor = conf["conv_expansion_factor"],
                                conv_kernel_size = conf["conv_kernel_size"],
                                attn_dropout = conf["attn_dropout"],
                                ff_dropout = conf["ff_dropout"],
                                conv_dropout = conf["conv_dropout"]
                            )
    return conformer_block

class LSTM_Projection(nn.Module):
    def __init__(self, input_size, hidden_size, linear_dim, num_layers=1, bidirectional=True, dropout=0):
        super(LSTM_Projection, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.forward_projection = nn.Linear(hidden_size, linear_dim)
        self.backward_projection = nn.Linear(hidden_size, linear_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, nframes):
        '''
        x: [batchsize, Time, Freq]
        nframes: [len_b1, len_b2, ..., len_bN]
        '''
        packed_x = nn.utils.rnn.pack_padded_sequence(x, nframes, batch_first=True)
        packed_x_1, hidden = self.LSTM(packed_x)
        x_1, l = nn.utils.rnn.pad_packed_sequence(packed_x_1, batch_first=True)
        forward_projection = self.relu(self.forward_projection(x_1[..., :self.hidden_size]))
        backward_projection = self.relu(self.backward_projection(x_1[..., self.hidden_size:]))
        # x_2: [batchsize, Time, linear_dim*2]
        x_2 = torch.cat((forward_projection, backward_projection), dim=2)
        return x_2


class CNN2D_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CNN2D_BN_Relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels) #(N,C,H,W) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True):
        super(SeparableConv1d,self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, kernel_size//2, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels) #(N,C,L) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MA_MSE(nn.Module):
    def __init__(self, fea_dim=20*128, n_heads=8, speaker_embedding_path=""):
        super(MA_MSE, self).__init__()

        self.n_heads = n_heads

        #Dictionary number of cluster * speaker_embedding_dim
        self.m = torch.from_numpy(np.load(speaker_embedding_path).astype(np.float32))
        self.N_clusters, Emb_dim = self.m.shape

        # Define matrices W (from audio feature) and U (from embedding)
        self.W = nn.Linear(fea_dim, n_heads)
        self.U = nn.Linear(Emb_dim, n_heads)

        self.v = nn.Linear(n_heads, 1)
    
    def forward(self, x, mask):
        '''
        x: Batch * Fea * Time
        mask: Batch * speaker * Time
        '''
        Batch, Fea, Time = x.shape
        num_speaker = mask.shape[1]
        #x_1: [Batch, num_speaker, Time, Fea]
        x_1 = x.repeat(1, num_speaker, 1).reshape(Batch, num_speaker, Fea, Time).transpose(2, 3)

        #x_2: Average [Batch, num_speaker, Fea]
        x_2 = torch.sum(x_1 * mask[..., None], axis=2) / (1e-10 + torch.sum(mask, axis=2)[..., None])

        #self.W(x_2) [Batch, num_speaker, n_heads]
        w = self.W(x_2).repeat(1, self.N_clusters, 1).reshape(Batch, self.N_clusters, num_speaker, self.n_heads).transpose(1, 2)

        #self.U(self.m) [N_clusters, n_heads]
        m = self.m.cuda()
        u = self.U(m).repeat(Batch*num_speaker, 1).reshape(Batch, num_speaker, self.N_clusters, self.n_heads)

        #c: Attention [Batch, num_speaker, N_clusters]
        c = self.v(torch.tanh(w + u)).squeeze(dim=3)

        #a: normalized attention values [Batch, num_speaker, N_clusters]
        a = torch.sigmoid(c)

        #e: weighted sum of the vectors [Batch, num_speaker, Emb_dim]
        #[Batch, num_speaker, N_clusters, 1] * [1, 1, N_clusters, Emb_dim]
        e = torch.sum(a[..., None] * m[None, None, ...], dim=2)

        return e
    

class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src


class CrossAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(src, tar, tar, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src


class Attention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, input):
        src, tar = input[0], input[1]
        #print(src.shape, tar.shape)
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(src, tar, tar, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.activation(self.linear1(src))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        
        return (src, src)


class CrossAttention_MC_MAMSE(nn.Module):

    def __init__(self, configs):
        super(CrossAttention_MC_MAMSE, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # MA-MSE
        self.mamse1 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads1"], speaker_embedding_path=configs["embedding_path1"])
        self.mamse2 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads2"], speaker_embedding_path=configs["embedding_path2"])
        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.Attention = CrossAttention(self.Linear_Shared_layer2_dim * 2, configs["n_head"], configs["dropout"])

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, overall_embedding, mask, nframes, return_embedding=False, split_seg=-1):
        '''
        x: Batch * CH * Freq * Time
        mask: Batch * speaker * Time
        overall_embedding: Batch * CH * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, ch, Freq, Time = x.shape
        _, _, num_speaker, _ = overall_embedding.shape
        
        x = x.reshape(batchsize*ch, Freq, Time)
        overall_embedding = overall_embedding.reshape(batchsize*ch, num_speaker, -1)
        mask = mask.unsqueeze(1).repeat(1, ch, 1, 1).reshape(batchsize*ch, num_speaker, Time)
        real_batchsize = batchsize
        batchsize = batchsize * ch

        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x = torch.cat((x, self.average_pooling(x)), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, Freq, Time] => [batchsize, Conv-4-out-filters, Freq, Time]
        x = self.Conv2d_SD(x)
        #print(x_5.shape)
        #[batchsize, Conv-4-out-filters, Freq, Time] => [ batchsize, Conv-4-out-filters*Freq, Time ]
        x = x.reshape(batchsize, -1, Time)
        Freq = x.shape[1]
        
        embedding1 = self.mamse1(x, mask) # [Batch, num_speaker, Emb_dim]
        embedding2 = self.mamse2(x, mask) # [Batch, num_speaker, Emb_dim]

        #*********************************************need to check***************** ***
        #print(x_1.repeat(1, speaker, 1).shape)
        x = x.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding1.shape[2]
        embedding1 = embedding1.reshape(-1, embedding_dim1)[..., None].expand(batchsize * self.output_speaker, embedding_dim1, Time)

        embedding_dim2 = embedding2.shape[2]
        embedding2 = embedding2.reshape(-1, embedding_dim2)[..., None].expand(batchsize * self.output_speaker, embedding_dim2, Time)

        overall_embedding_dim = overall_embedding.shape[2]
        overall_embedding = overall_embedding.reshape(-1, overall_embedding_dim)[..., None].expand(batchsize * self.output_speaker, overall_embedding_dim, Time)
        #print(embedding_reshape.shape)
        x = torch.cat((x, embedding1, embedding2, overall_embedding), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x = self.relu(self.Linear(x.transpose(1, 2)))

        lens = [ n for n in nframes for i in range(self.output_speaker) for c in range(ch) ] 
        x = self.Shared_BLSTMP_1(x, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)  => (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x = self.Shared_BLSTMP_2(x, lens)
        # B * C * S * T * F
        x_reshape = x.reshape(real_batchsize, ch, num_speaker, Time, -1)
        x_reshape = (torch.sum(x_reshape, 1, keepdim=True) - x_reshape) / (ch - 1)
        x = self.Attention(x, x_reshape.reshape(real_batchsize*ch*num_speaker, Time, -1)).reshape(real_batchsize, ch, num_speaker, Time, -1)
        x = torch.mean(x, axis=1)
        batchsize = real_batchsize
        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        x = x.transpose(1, 2).reshape(batchsize, Time, -1)
        lens = nframes
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x = self.BLSTMP(x, lens)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x = x.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x))
        return out


class CrossAttention_MC_MAMSE_Conformer(nn.Module):

    def __init__(self, configs):
        super(CrossAttention_MC_MAMSE_Conformer, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]

        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # MA-MSE
        self.mamse1 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads1"], speaker_embedding_path=configs["embedding_path1"])
        self.mamse2 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads2"], speaker_embedding_path=configs["embedding_path2"])
        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, configs["share_conformer"]["dim"])
        self.relu = nn.ReLU(True)

        self.Share_Conformer = nn.Sequential()
        for i in range(configs["share_conformer_layers"]):
            self.Share_Conformer.add_module(f'Share_Conformer{i}', make_conformer_block(configs["share_conformer"]))
        
        self.Attention = CrossAttention(configs["share_conformer"]["dim"], configs["n_head"], configs["dropout"])

        conbine_speaker_size = configs["share_conformer"]["dim"] * configs["output_speaker"]
        self.combine_Linear = nn.Linear(conbine_speaker_size, configs["conformer"]["dim"])

        self.Conformer = nn.Sequential()
        for i in range(configs["conformer_layers"]):
            self.Conformer.add_module(f'Conformer{i}', make_conformer_block(configs["conformer"]))
        FC = {}
        for i in range(configs["output_speaker"]):
            FC[str(i)] = nn.Linear(configs["conformer"]["dim"], 2)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, overall_embedding, mask, nframes, return_embedding=False, split_seg=-1):
        '''
        x: Batch * CH * Freq * Time
        mask: Batch * speaker * Time
        overall_embedding: Batch * CH * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, ch, Freq, Time = x.shape
        _, _, num_speaker, _ = overall_embedding.shape
        
        x = x.reshape(batchsize*ch, Freq, Time)
        overall_embedding = overall_embedding.reshape(batchsize*ch, num_speaker, -1)
        mask = mask.unsqueeze(1).repeat(1, ch, 1, 1).reshape(batchsize*ch, num_speaker, Time)
        real_batchsize = batchsize
        batchsize = batchsize * ch

        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x = torch.cat((x, self.average_pooling(x)), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, Freq, Time] => [batchsize, Conv-4-out-filters, Freq, Time]
        x = self.Conv2d_SD(x)
        #print(x_5.shape)
        #[batchsize, Conv-4-out-filters, Freq, Time] => [ batchsize, Conv-4-out-filters*Freq, Time ]
        x = x.reshape(batchsize, -1, Time)
        Freq = x.shape[1]
        
        embedding1 = self.mamse1(x, mask) # [Batch, num_speaker, Emb_dim]
        embedding2 = self.mamse2(x, mask) # [Batch, num_speaker, Emb_dim]

        #*********************************************need to check***************** ***
        #print(x_1.repeat(1, speaker, 1).shape)
        x = x.repeat(1, num_speaker, 1).reshape(batchsize * num_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding1.shape[2]
        embedding1 = embedding1.reshape(-1, embedding_dim1)[..., None].expand(batchsize * num_speaker, embedding_dim1, Time)

        embedding_dim2 = embedding2.shape[2]
        embedding2 = embedding2.reshape(-1, embedding_dim2)[..., None].expand(batchsize * num_speaker, embedding_dim2, Time)

        overall_embedding_dim = overall_embedding.shape[2]
        overall_embedding = overall_embedding.reshape(-1, overall_embedding_dim)[..., None].expand(batchsize * num_speaker, overall_embedding_dim, Time)
        #print(embedding_reshape.shape)
        x = torch.cat((x, embedding1, embedding2, overall_embedding), dim=1)

        x = self.relu(self.Linear(x.transpose(1, 2)))
        x = self.Share_Conformer(x)

        # B * C * S * T * F
        x_reshape = x.reshape(real_batchsize, ch, num_speaker, Time, -1)
        x_reshape = (torch.sum(x_reshape, 1, keepdim=True) - x_reshape) / (ch - 1)
        x = self.Attention(x, x_reshape.reshape(real_batchsize*ch*num_speaker, Time, -1)).reshape(real_batchsize, ch, num_speaker, Time, -1)
        x = torch.mean(x, axis=1)
        batchsize = real_batchsize

        x = x.transpose(1, 2).reshape(batchsize, Time, -1)
        x = self.relu(self.combine_Linear(x))

        x = self.Conformer(x)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x = x.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x))
        return out


class WavLM_CrossAttention_MC_MAMSE(nn.Module):

    def __init__(self, configs):
        super(WavLM_CrossAttention_MC_MAMSE, self).__init__()

        checkpoint = torch.load(configs["wavlm_pt"])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.wavlm = WavLM(cfg)
        self.wavlm.load_state_dict(checkpoint['model'])
        self.layer_weight = nn.Parameter(torch.zeros(self.wavlm.cfg.encoder_layers+1))
        for _, param in self.wavlm.named_parameters():
            param.requires_grad = False
        
        self.SD = make_attention_layer(d_model=1024, nhead=8, num_layers=3)

        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.mamse1 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads1"], speaker_embedding_path=configs["embedding_path1"])
        self.mamse2 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads2"], speaker_embedding_path=configs["embedding_path2"])
        # Speaker Detection Block
        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.Attention = CrossAttention(self.Linear_Shared_layer2_dim * 2, configs["n_head"], configs["dropout"])

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, overall_embedding, mask, nframes, return_embedding=False, split_seg=-1):
        '''
        x: Batch * CH * Freq * Time
        mask: Batch * speaker * Time
        overall_embedding: Batch * CH * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        real_batchsize, ch, Time = x.shape
        batchsize = real_batchsize * ch
        x = x.reshape(batchsize, Time)
        x = torch.nn.functional.layer_norm(x, x.shape)
        self.wavlm.eval()
        with torch.no_grad():
            _, layer_results = self.wavlm.extract_features(x, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
        x = torch.cat([x.detach().unsqueeze(1) for x, _ in layer_results], dim=1)
        norm_weights = F.softmax(self.layer_weight, dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = (x * norm_weights).sum(1) # T B F
        x = x.permute(1, 0, 2)
        Time = x.shape[1]
        #print(x.shape)
        x = self.SD((x, x))[0] # B T 512

        _, _, num_speaker, _ = overall_embedding.shape
        
        x = x.transpose(1, 2)
        Freq = x.shape[1]
        overall_embedding = overall_embedding.reshape(real_batchsize*ch, num_speaker, -1)
        mask = mask.unsqueeze(1).repeat(1, ch, 1, 1).reshape(real_batchsize*ch, num_speaker, Time)

        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())
        
        embedding1 = self.mamse1(x, mask) # [Batch, num_speaker, Emb_dim]
        embedding2 = self.mamse2(x, mask) # [Batch, num_speaker, Emb_dim]

        #*********************************************need to check***************** ***
        x = x.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding1.shape[2]
        embedding1 = embedding1.reshape(-1, embedding_dim1)[..., None].expand(batchsize * self.output_speaker, embedding_dim1, Time)

        embedding_dim2 = embedding2.shape[2]
        embedding2 = embedding2.reshape(-1, embedding_dim2)[..., None].expand(batchsize * self.output_speaker, embedding_dim2, Time)

        overall_embedding_dim = overall_embedding.shape[2]
        overall_embedding = overall_embedding.reshape(-1, overall_embedding_dim)[..., None].expand(batchsize * self.output_speaker, overall_embedding_dim, Time)
        #print(embedding_reshape.shape)
        x = torch.cat((x, embedding1, embedding2, overall_embedding), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x = self.relu(self.Linear(x.transpose(1, 2)))

        lens = [ n for n in nframes for i in range(self.output_speaker) for c in range(ch) ] 
        x = self.Shared_BLSTMP_1(x, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)  => (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x = self.Shared_BLSTMP_2(x, lens)
        # B * C * S * T * F
        x_reshape = x.reshape(real_batchsize, ch, num_speaker, Time, -1)
        x_reshape = (torch.sum(x_reshape, 1, keepdim=True) - x_reshape) / (ch - 1)
        x = self.Attention(x, x_reshape.reshape(real_batchsize*ch*num_speaker, Time, -1)).reshape(real_batchsize, ch, num_speaker, Time, -1)
        x = torch.mean(x, axis=1)
        batchsize = real_batchsize
        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        x = x.transpose(1, 2).reshape(batchsize, Time, -1)
        lens = nframes
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x = self.BLSTMP(x, lens)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x = x.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x))
        return out


class WavLM_CrossAttention_MC_MAMSE2(nn.Module):

    def __init__(self, configs):
        super(WavLM_CrossAttention_MC_MAMSE2, self).__init__()

        checkpoint = torch.load(configs["wavlm_pt"])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.wavlm = WavLM(cfg)
        self.wavlm.load_state_dict(checkpoint['model'])
        self.layer_weight = nn.Parameter(torch.zeros(self.wavlm.cfg.encoder_layers+1))
        for _, param in self.wavlm.named_parameters():
            param.requires_grad = False
        
        self.SD = make_attention_layer(d_model=1024, nhead=8, num_layers=3)

        self.Attention = CrossAttention(1024, configs["n_head"], configs["dropout"])

        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.mamse1 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads1"], speaker_embedding_path=configs["embedding_path1"])
        self.mamse2 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads2"], speaker_embedding_path=configs["embedding_path2"])
        # Speaker Detection Block
        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, overall_embedding, mask, nframes, return_embedding=False, split_seg=-1):
        '''
        x: Batch * CH * Freq * Time
        mask: Batch * speaker * Time
        overall_embedding: Batch * CH * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        real_batchsize, ch, Time = x.shape
        batchsize = real_batchsize * ch
        x = x.reshape(batchsize, Time)
        x = torch.nn.functional.layer_norm(x, x.shape)
        self.wavlm.eval()
        with torch.no_grad():
            _, layer_results = self.wavlm.extract_features(x, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
        x = torch.cat([x.detach().unsqueeze(1) for x, _ in layer_results], dim=1)
        norm_weights = F.softmax(self.layer_weight, dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = (x * norm_weights).sum(1) # T B F
        x = x.permute(1, 0, 2)
        Time = x.shape[1]
        #print(x.shape)
        x = self.SD((x, x))[0] # B T 512

        x_reshape = x.reshape(real_batchsize, ch, Time, -1)
        x_reshape = (torch.sum(x_reshape, 1, keepdim=True) - x_reshape) / (ch - 1)
        x = self.Attention(x, x_reshape.reshape(real_batchsize*ch, Time, -1)).reshape(real_batchsize, ch, Time, -1)
        x = torch.mean(x, axis=1)

        _, _, num_speaker, _ = overall_embedding.shape
        
        x = x.transpose(1, 2)
        Freq = x.shape[1]
        overall_embedding = torch.mean(overall_embedding, axis=1)
        
        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())
        
        embedding1 = self.mamse1(x, mask) # [Batch, num_speaker, Emb_dim]
        embedding2 = self.mamse2(x, mask) # [Batch, num_speaker, Emb_dim]
        batchsize = real_batchsize
        #*********************************************need to check***************** ***
        x = x.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding1.shape[2]
        embedding1 = embedding1.reshape(-1, embedding_dim1)[..., None].expand(batchsize * self.output_speaker, embedding_dim1, Time)

        embedding_dim2 = embedding2.shape[2]
        embedding2 = embedding2.reshape(-1, embedding_dim2)[..., None].expand(batchsize * self.output_speaker, embedding_dim2, Time)

        overall_embedding_dim = overall_embedding.shape[2]
        overall_embedding = overall_embedding.reshape(-1, overall_embedding_dim)[..., None].expand(batchsize * self.output_speaker, overall_embedding_dim, Time)
        #print(embedding_reshape.shape)
        x = torch.cat((x, embedding1, embedding2, overall_embedding), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x = self.relu(self.Linear(x.transpose(1, 2)))

        lens = [ n for n in nframes for i in range(self.output_speaker) ] 
        x = self.Shared_BLSTMP_1(x, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)  => (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x = self.Shared_BLSTMP_2(x, lens)
        # B * C * S * T * F
        
        batchsize = real_batchsize
        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        x = x.transpose(1, 2).reshape(batchsize, Time, -1)
        lens = nframes
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x = self.BLSTMP(x, lens)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x = x.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x))
        return out


class WavLM_Transformer_CrossAttention_MC_MAMSE(nn.Module):

    def __init__(self, configs):
        super(WavLM_Transformer_CrossAttention_MC_MAMSE, self).__init__()

        checkpoint = torch.load(configs["wavlm_pt"])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.wavlm = WavLM(cfg)
        self.wavlm.load_state_dict(checkpoint['model'])
        self.layer_weight = nn.Parameter(torch.zeros(self.wavlm.cfg.encoder_layers+1))
        for _, param in self.wavlm.named_parameters():
            param.requires_grad = False
        
        # MA-MSE
        self.mamse1 = MA_MSE(fea_dim=512, n_heads=8, speaker_embedding_path=configs["embedding_path1"])
        # Speaker Detection Block
        self.FC_1 = nn.Linear(1024, 512)
        self.SD_1 = make_attention_layer(d_model=512, nhead=8, num_layers=3)
        self.FC_2 = nn.Linear(512+100+100, 512)
        self.SD_2 = make_attention_layer(d_model=512, nhead=8, num_layers=3)
        self.CrossAttention_Channel = make_attention_layer(d_model=512, nhead=8)
        self.SD_3 = make_attention_layer(d_model=512, nhead=8, num_layers=3)
        self.CrossAttention_Speaker = make_attention_layer(d_model=512, nhead=8)
        self.SD_4 = make_attention_layer(d_model=512, nhead=8, num_layers=3)
        
        self.FC = nn.Linear(512, 2)

    def forward(self, x, overall_embedding, mask, nframes, return_embedding=False, split_seg=-1):
        '''
        x: Batch * CH * Time
        mask: Batch * speaker * Time
        overall_embedding: Batch * CH * speaker(4) * Embedding
        '''
        real_batchsize, ch, Time = x.shape
        batchsize = real_batchsize * ch
        _, _, speaker, _ = overall_embedding.shape
        x = x.reshape(batchsize, Time)
        x = torch.nn.functional.layer_norm(x, x.shape)
        self.wavlm.eval()
        with torch.no_grad():
            _, layer_results = self.wavlm.extract_features(x, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
        x = torch.cat([x.detach().unsqueeze(1) for x, _ in layer_results], dim=1)
        norm_weights = F.softmax(self.layer_weight, dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = (x * norm_weights).sum(1) # T B F
        x = x.permute(1, 0, 2)
        Time = x.shape[1]
        x = self.FC_1(x) # B T 512
        #print(x.shape)
        x = self.SD_1((x, x))[0] # B T 512
        x_reshape = x.repeat(1, speaker, 1).reshape(batchsize * speaker, Time, -1)
        x = x.transpose(1, 2)
        mask = mask.unsqueeze(1).repeat(1, ch, 1, 1).reshape(batchsize, speaker, Time)

        embedding1 = self.mamse1(x, mask)
        embedding_dim1 = embedding1.shape[2]
        embedding_reshape1 = embedding1.repeat(1, 1, Time).reshape(batchsize * speaker, Time, embedding_dim1)
        overall_embedding_dim = overall_embedding.shape[3]
        overall_embedding_reshape = overall_embedding.repeat(1, 1, 1, Time).reshape(batchsize * speaker, Time, overall_embedding_dim)

        #print(x.shape, embedding_reshape1.shape, overall_embedding_reshape.shape)
        x = torch.cat((x_reshape, embedding_reshape1, overall_embedding_reshape), dim=2)

        x = self.FC_2(x) # B T 512
        
        x = self.SD_2((x, x))[0]

        x_reshape = x.reshape(real_batchsize, ch, speaker, Time, -1)
        x_reshape = (torch.sum(x_reshape, 1, keepdim=True) - x_reshape) / (ch - 1)
        x = self.CrossAttention_Channel((x, x_reshape.reshape(real_batchsize*ch*speaker, Time, -1)))[0]
        x = torch.mean(x.reshape(real_batchsize, ch, speaker, Time, -1), axis=1).reshape(real_batchsize*speaker, Time, -1) # B S T F

        x = self.SD_3((x, x))[0]

        x_reshape = x.reshape(real_batchsize, speaker, Time, -1)
        x_reshape = (torch.sum(x_reshape, 1, keepdim=True) - x_reshape) / (speaker - 1)
        x = self.CrossAttention_Speaker((x, x_reshape.reshape(real_batchsize*speaker, Time, -1)))[0]

        x = self.SD_4((x, x))[0]
        
        x = self.FC(x).reshape(real_batchsize, speaker, Time, -1)

        out = []
        for s in range(speaker):
            out.append(x[:, s, ...].reshape(real_batchsize*Time, -1))
        return out