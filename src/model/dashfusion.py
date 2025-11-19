import torch
from torch import nn
import numpy as np

from model.text_encoder import TextEncoder
from model.vision_encoder import VisionEncoder
from model.audio_encoder import AudioEncoder
from model.layers import TemporalAlignment, HierarchicalBottleneckFusion
from model.MLP import BaseClassifier, FeatureProjector
import config as default_config
from utils import check_dir, cont_NTXentLoss

class DashFusion(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, d_model = None, depth = None, drop_out=None, config=default_config):
        super(DashFusion, self).__init__()
        self.config = config
        if encoder_fea_dim is None:
            encoder_fea_dim = config.PARAM.downStream.encoder_fea_dim
        if d_model is None:
            d_model = config.PARAM.downStream.d_model
        if drop_out is None:
            drop_out = config.PARAM.downStream.drop_out
        if depth is None:
            depth = config.PARAM.downStream.fusion_depth

        # feature extractor
        self.text_encoder = TextEncoder(config=config)
        self.vision_encoder = VisionEncoder(config=config)
        self.audio_encoder = AudioEncoder(config=config)

        # feature alignment
        self.temporal_alignment = TemporalAlignment(encoder_fea_dim, 8, 32, drop_out)

        # Feature Fusion
        self.HBF = HierarchicalBottleneckFusion(d_model=d_model, n_heads=8, ff_dim=d_model*2, drop_out=drop_out, depth=depth)

        hidden_size =  [d_model*2, d_model]
        self.classifier = BaseClassifier(input_size=d_model*4, hidden_size=hidden_size, output_size=1)

        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.heat = config.PARAM.downStream.heat
        self.ntxent_loss = cont_NTXentLoss(temperature=self.heat)
        self.set_train()

    def forward(self, sample1, sample2, return_loss=True, device=None):
        if device is None:
            device = self.device

        text1 = sample1['raw_text']
        vision1 = sample1['vision'].clone().detach().to(device).float()
        audio1 = sample1['audio'].clone().detach().to(device).float()
        label1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        key_padding_mask_V1, key_padding_mask_A1 = (sample1['vision_padding_mask'].clone().detach().to(device),
                                                    sample1['audio_padding_mask'].clone().detach().to(device))
        
        t_embed_all, t_embed = self.text_encoder(text1, device=device)
        a_embed_all, a_embed = self.audio_encoder(audio1, key_padding_mask=key_padding_mask_A1, device=device) #.squeeze()
        v_embed_all, v_embed = self.vision_encoder(vision1, key_padding_mask=key_padding_mask_V1, device=device) #.squeeze()

        
        # no return loss，update similarity matrix
        if not return_loss:
            return (t_embed, v_embed, a_embed)

        # retutn loss，complete training
        else:

            ### Dual-stream Alignment
            # Temporal Alignment
            t_a_v_embed_all = self.temporal_alignment(t_embed_all, a_embed_all, v_embed_all)
            t_a_v_embed = t_a_v_embed_all[:, 0]

            # Semantic Alignment
            const_loss = 0
            # sample2: construct contrastive pair for contrastive learning, 1 sample1 with 6 sample2(2 positive, 4 negative)
            if sample2 is not None:
                text2 = sample2['raw_text']
                vision2 = sample2['vision'].clone().detach().to(device).float()
                audio2 = sample2['audio'].clone().detach().to(device).float()
                label2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
                key_padding_mask_V2, key_padding_mask_A2 = (sample2['vision_padding_mask'].clone().detach().to(device),
                                                            sample2['audio_padding_mask'].clone().detach().to(device))

                t2_embed_all, t2_embed = self.text_encoder(text2, device=device)
                a2_embed_all, a2_embed = self.audio_encoder(audio2, key_padding_mask=key_padding_mask_A2, device=device)
                v2_embed_all, v2_embed = self.vision_encoder(vision2, key_padding_mask=key_padding_mask_V2, device=device)
                

                # Temporal Alignment
                t_a_v2_embed_all = self.temporal_alignment(t2_embed_all, a2_embed_all, v2_embed_all)
                t_a_v2_embed = t_a_v2_embed_all[:, 0]

                # pre_sample_x：[T,T1,T2,T3,T4,T5,T6,,V,V1,...V6,A,A1,...,A6,....,F1,F2,...,F6]
                t1, p, t2, n = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0, 0, 7, 7, 14, 14, 21, 21,], device=device), \
                                torch.tensor([7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20, 1, 2, 8, 9, 15, 16, 22, 23,], device=device), \
                                torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14, 21, 21, 21, 21,], device=device), \
                                torch.tensor([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20, 24, 25, 26, 27,], device=device)
                
                
                
                indices_tuple = (t1, p, t2, n)
                # Define a label for each sample. Each group has 7 samples: 1 is the archor, 2-3 are positive samples, and 4-7 are negative samples.
                pre_sample_label = torch.tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,]) 
                for i in range(len(t_embed)):
                    pre_sample_x = []
                    for fea1, fea2 in zip([t_embed, v_embed, a_embed, t_a_v_embed], [t2_embed, v2_embed, a2_embed, t_a_v2_embed]):
                        pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[6 * i:6 * (i + 1)]), dim=0))
                    
                    pre_sample_x = torch.cat(pre_sample_x, dim=0)
                    const_loss += self.ntxent_loss(pre_sample_x, pre_sample_label, indices_tuple=indices_tuple)
                const_loss /= len(t_embed)
                
            ### Hierarchical Bottleneck Fusion
            pred_embed = self.HBF(t_a_v_embed_all, t_embed_all, v_embed_all, a_embed_all)

            ### Classifier
            pred = self.classifier(pred_embed).squeeze(-1)
            pred_loss = self.criterion(pred, label1)
            
            loss = pred_loss + 0.2*const_loss

            return pred, loss, pred_loss, const_loss

    def save_model(self, name, dataset):
        # save all modules
        if dataset == 'mosi':
            model_path = self.config.MOSI.model_path + name + '_model.ckpt'
        elif dataset == 'mosei':
            model_path = self.config.MOSEI.model_path + name + '_model.ckpt'
        elif dataset == 'sims':
            model_path = self.config.SIMS.model_path + name + '_model.ckpt'
        print('model saved at:')
        print(model_path)
        torch.save(self.state_dict(), model_path)

    def load_model(self, name, dataset):

        if dataset == 'mosi':
            model_path = self.config.MOSI.model_path + name + '_model.ckpt'
        elif dataset == 'mosei':
            model_path = self.config.MOSEI.model_path + name + '_model.ckpt'
        elif dataset == 'sims':
            model_path = self.config.SIMS.model_path + name + '_model.ckpt'
        print('model loaded from:')
        print(model_path)
        self.load_state_dict(torch.load(model_path, map_location=self.device))

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True, True, True]

        for param in self.parameters():
            param.requires_grad = train_module[3]
        self.text_encoder.set_train(train_module=train_module[0:2])
        self.vision_encoder.set_train(train_module=train_module[2])
        self.audio_encoder.set_train(train_module=train_module[2])

