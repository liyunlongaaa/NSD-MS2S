# -*- coding: utf-8 -*-

configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"conformer_layers": 6,
"conformer_conv_kernel_size": 15,
"conformer_ff_dropout": 0.1,
"decoder_layers": 6,
"decoder_num_heads": 8,
"decoder_ffn_num_hiddens": 1024,
"decoder_mlp_num_hiddens": 512,
"decoder_attn_dropout": 0.0,
"decoder_dropout": 0.0,
"decode_Time": 800,
"fea_dim": 512,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"ma_mse_layers_1":1,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"ma_mse_layers_2":1,
"output_speaker": 4
}

configs3_2Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"conformer_layers": 6,
"conformer_conv_kernel_size": 15,
"conformer_ff_dropout": 0.1,
"decoder_layers": 6,
"decoder_num_heads": 8,
"decoder_ffn_num_hiddens": 1024,
"decoder_mlp_num_hiddens": 512,
"decoder_attn_dropout": 0.0,
"decoder_dropout": 0.0,
"decode_Time": 800,
"fea_dim": 512,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"ma_mse_layers_1":3,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"ma_mse_layers_2":3,
"output_speaker": 2
}

configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM_MOE = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"moe_geglu":False,
"moe_layer":[5],
"lambda_entropy": 0.5,                                                                 ''
"decoder_ffn_num_hiddens":1024,
"conformer_layers": 6,
"conformer_conv_kernel_size": 15,
"conformer_ff_dropout": 0.1,
"decoder_layers": 6,
"decoder_num_heads": 8,
"decoder_mlp_num_hiddens": 512,
"decoder_attn_dropout": 0.0,
"decoder_dropout": 0.0,
"decode_Time": 800,
"fea_dim": 512,
"num_slots": 4,
"num_experts": 6,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"ma_mse_layers_1":1,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"ma_mse_layers_2":1,
"output_speaker": 4
}

configs = {
    "configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM": configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM,
    "configs3_2Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM": configs3_2Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM,
    "configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM_MOE": configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM_MOE
}
