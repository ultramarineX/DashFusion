import torch
import torch.nn as nn
import config as default_config
from model.MLP import FeatureProjector


class PositionEncodingTraining(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """
    def __init__(self, fea_size=None, tf_hidden_dim=None, drop_out=None, dataset=None, config=default_config):
        super().__init__()
        if dataset is None:
            dataset = config.dataset
        if tf_hidden_dim is None:
            tf_hidden_dim = config.PARAM.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.PARAM.downStream.v_drop_out

        if dataset == 'mosi':
            num_patches = 500
            if fea_size is None:
                fea_size = config.MOSI.vision_fea_dim
        elif dataset == 'mosei':
            num_patches = 500
            if fea_size is None:
                fea_size = config.MOSEI.vision_fea_dim
        elif dataset  == 'sims':
            num_patches = 55
            if fea_size is None:
                fea_size = config.SIMS.vision_fea_dim

        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TfEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu',
                 config=default_config):
        super(TfEncoder, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')

        self.device = config.DEVICE
        self.model_type = 'vision_encoder'
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining()

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        return output.transpose(0, 1)


class VisionEncoder(nn.Module):
    def __init__(self, name=None, fea_size=None, encoder_fea_dim=None, proj_fea_dim=None, nhead=None, dim_feedforward=None, 
                dataset=None,num_layers=None, drop_out=0.5, config=default_config):
        super(VisionEncoder, self).__init__()
        self.name = name
        self.with_projector=False
        if dataset is None:
            dataset = config.dataset
        if encoder_fea_dim is None:
            encoder_fea_dim = config.PARAM.downStream.encoder_fea_dim
        if proj_fea_dim is None:
            proj_fea_dim = config.PARAM.downStream.d_model
        if nhead is None:
            nhead = config.PARAM.downStream.vision_nhead
        if drop_out is None:
            drop_out = config.PARAM.downStream.v_drop_out
        if dim_feedforward is None:
            dim_feedforward = config.PARAM.downStream.encoder_fea_dim * 2
        if num_layers is None:
            num_layers = config.PARAM.downStream.vision_tf_num_layers
        if fea_size is None:
            if dataset == 'mosi':
                fea_size = config.MOSI.vision_fea_dim   
            elif dataset == 'mosei':
                fea_size = config.MOSEI.vision_fea_dim
            elif dataset  == 'sims':
                fea_size = config.SIMS.vision_fea_dim
        if encoder_fea_dim != proj_fea_dim:
            self.with_projector = True
            self.projector = FeatureProjector(encoder_fea_dim, proj_fea_dim, drop_out=drop_out, config=config)

        self.fc = nn.Linear(fea_size, encoder_fea_dim)
        self.encoder = TfEncoder(d_model=encoder_fea_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                 num_layers=num_layers,dropout=drop_out, activation='gelu',config=config)

        self.device = config.DEVICE
        self.encoder.device = self.device
        self.activation = nn.Tanh()
        self.cls_embedding = nn.Parameter()
        self.layernorm = nn.LayerNorm(encoder_fea_dim)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)
        
    def forward(self, vision, key_padding_mask, device=None):
        if device is None:
            device = self.device

        x = self.encoder(vision, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        if self.with_projector:
            x = self.projector(x)
        mask_expanded = (~key_padding_mask).unsqueeze(-1).expand(x.size()).float()
        x = x * mask_expanded
        x_avc = x.sum(dim=1) / mask_expanded.sum(dim=1)
        return x, x_avc

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = True
        for param in self.parameters():
            param.requires_grad = train_module
