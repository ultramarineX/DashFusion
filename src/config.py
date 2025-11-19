import torch
import os
from utils import check_dir

#seed = [1, 12, 123, 1234]
seed = [12]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = 'sims'   # mosi, mosei, sims
USEROBERTA = False


class PARAM:

    class downStream:
        model_path = 'ckpt'
        metric = 'MAE'  # Has0_acc_2, Non0_acc_2
        load_metric = 'best_' + metric
        check_list = [metric]
        check = {metric: 10000 if metric == 'Loss' or metric == 'MAE' else 0}
        use_reg = True

        encoder_fea_dim = 128
        text_fea_dim = 768
        d_model = 128
        drop_out = 0.0
        t_drop_out = 0.1
        a_drop_out = 0.1
        v_drop_out = 0.1
        vision_nhead = 8
        audio_nhead = 8
        heat = 0.5


        if dataset == 'mosei':
            vision_tf_num_layers = 4  ## 4 for MOSEI
            audio_tf_num_layers = 4  ## 4 for MOSEI
            fusion_depth = 3  ## 3 for MOSEI
        else:
            vision_tf_num_layers = 2  ## 2 for MOSI, CH-SIMS
            audio_tf_num_layers = 2  ## 2 for MOSI, CH-SIMS
            fusion_depth = 2  ## 2 for MOSI, CH-SIMS
    
    
    class Train:
        if dataset == 'mosei':
            batch_size = 8  ## 8 for MOSEI (24G memory)
            lr = 2e-5  ## 2e-5 for MOSEI 
            epoch = 25  ## 125 for MOSEI
            decay = 1e-3
            num_warm_up = 1  ## 1 for MOSEI
        else:
            batch_size = 16  ## 16 for MOSI, CH-SIMS (24G memory)
            lr = 5e-5  ## 5e-5 for MOSI, CH-SIMS
            epoch = 4  ## 100 for MOSI, CH-SIMS
            decay = 1e-3
            num_warm_up = 10  ## 10 for MOSI, CH-SIMS


class MOSI:
    raw_data_path = 'dataset/MOSI/Processed/unaligned_50.pkl'
    model_path = 'ckpt/MOSI/'
    result_path = 'result/MOSI/'
    log_path = 'log/MOSI/'
    vision_fea_dim = 20
    video_seq_len = 500
    audio_fea_dim = 5
    audio_seq_len = 375


class MOSEI:
    raw_data_path = 'dataset/MOSEI/Processed/unaligned_50.pkl'
    model_path = 'ckpt/MOSEI/'
    result_path = 'result/MOSEI/'
    log_path = 'log/MOSEI/'
    vision_fea_dim = 35
    video_seq_len = 500
    audio_fea_dim = 74
    audio_seq_len = 500

class SIMS:
    raw_data_path = 'dataset/SIMS/Processed/unaligned_39.pkl'
    model_path = 'ckpt/SIMS/'
    result_path = 'result/SIMS/'
    log_path = 'log/SIMS/'
    vision_fea_dim = 709
    video_seq_len = 55
    audio_fea_dim = 33
    audio_seq_len = 400

