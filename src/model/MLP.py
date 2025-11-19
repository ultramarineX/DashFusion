import torch.nn as nn
import torch
import config as default_config
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, drop_out=0.1, config=default_config):
        super(FeatureProjector, self).__init__()
        self.device = config.DEVICE 

        # 定义线性映射层的结构
        self.proj = nn.ModuleList()
        for i in range(num_layers):
            # 第一层的输入维度是 input_dim，输出维度是 project_size
            if i == 0:
                self.proj.append(nn.Linear(input_dim, output_dim, bias=False))
                #self.proj.append(nn.Dropout(p=drop_out))
                
            else :
                self.proj.append(nn.Linear(output_dim, output_dim, bias=False))
                #self.proj.append(nn.Dropout(p=drop_out))
            # 在每个线性映射层后添加 GELU 激活函数
            self.proj.append(nn.GELU())

        # 定义 layer normalization 层
        self.layernorm = nn.LayerNorm(output_dim)  # 对 MLP 输出做 layer normalization

        self.MLP = nn.Sequential(*self.proj)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        # 输入数据的形状为 [seq_length, batch_size, feature_dim]
        # 对输入数据进行 dropout 操作
        dropped = self.drop(x)
        x = self.MLP(x)
        x = self.layernorm(x)
        return x


class BaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name=None):
        super(BaseClassifier, self).__init__()
        self.name = name
        # ModuleList = [nn.Dropout(p=drop_out)]
        ModuleList = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                ModuleList.append(nn.Linear(input_size, h))
                ModuleList.append(nn.GELU())
            else:
                ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(nn.GELU())
        ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)

    def forward(self, x):
        x = self.MLP(x)
        return x
