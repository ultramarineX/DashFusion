from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
from model.MLP import FeatureProjector
import torch
from torch import nn
import config as default_config


class TextEncoder(nn.Module):
    def __init__(self, name=None, fea_size=None, proj_fea_dim=None, drop_out=None, config=default_config):
        super(TextEncoder, self).__init__()
        self.name = name
        self.with_projector=False
        if fea_size is None:
            fea_size = config.PARAM.downStream.text_fea_dim
        if proj_fea_dim is None:
            proj_fea_dim = config.PARAM.downStream.d_model
        if drop_out is None:
            drop_out = config.PARAM.downStream.t_drop_out
        if config.dataset=='sims':
            if config.USEROBERTA:
                self.tokenizer = RobertaTokenizer.from_pretrained('hfl/chinese-robert-wwm-ext')
                self.extractor = RobertaTokenizer.from_pretrained('hfl/chinese-robert-wwm-ext')
            else:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                self.extractor = BertModel.from_pretrained('bert-base-chinese')
        else:
            if config.USEROBERTA:
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                self.extractor = RobertaModel.from_pretrained('roberta-base')
            else:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.extractor = BertModel.from_pretrained('bert-base-uncased')
        if fea_size != proj_fea_dim:
            self.with_projector = True
            self.projector = FeatureProjector(input_dim=fea_size, output_dim=proj_fea_dim, drop_out=drop_out, config=config)
        self.device = config.DEVICE

    def forward(self, text, device=None):
        if device is None:
            device = self.device

        x = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        attention_mask = x['attention_mask']
        x = self.extractor(**x)
        x = x['last_hidden_state']
        #x_mean = x['pooler_output'] 
        if self.with_projector:
            x = self.projector(x)
        mask_expanded = (attention_mask).float().unsqueeze(-1).expand(x.size())
        x = x * mask_expanded
        x_avc = x.sum(dim=1) / mask_expanded.sum(dim=1)
        return x, x_avc

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        for name, param in self.extractor.named_parameters():
                param.requires_grad = train_module[0]

        if self.with_projector:
            for param in self.projector.parameters():
                param.requires_grad = train_module[1]
