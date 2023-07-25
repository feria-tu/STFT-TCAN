# '''
# 基于多变量的时间序列异常检测
# based on TCN with attention
# '''
import torch
import torch.nn as nn
from models.tcn import TemporalConvNet
import math
from torch.nn import TransformerEncoderLayer,TransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)

class WTTCAN(nn.Module):
    def __init__(self, feats,lr):
        super(WTTCAN,self).__init__()
        self.name = 'WTTCAN'
        # 根据不同的数据集选择不同的lr
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.tcn = TemporalConvNet(
            num_outputs=feats,kernel_size=3,dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        '''
        d_model：输入特征的维度（或称为模型的隐藏维度）。它决定了输入和输出的特征维度大小。
        nhead：多头自注意力机制中注意力头的数量。注意力头允许模型同时关注输入的不同部分，以捕捉更丰富的信息。
        dim_feedforward：前馈神经网络（Feed-Forward Network）中间层的维度大小。它是一个整数值，定义了Transformer模型中的前馈网络的隐藏层维度。
        dropout：用于控制模型的dropout概率。在训练过程中，dropout可以防止过拟合，增强模型的泛化能力。
        '''
        encoder_layer1 = TransformerEncoderLayer(
            d_model=feats,nhead=feats,dim_feedforward=16,dropout=0.1)
        encoder_layer2 = TransformerEncoderLayer(
            d_model=feats,nhead=feats,dim_feedforward=16,dropout=0.1)
        '''
        encoder_layer：TransformerEncoderLayer的实例，用于定义每个编码器层的结构和参数。
        num_layers：编码器层的数量，决定了模型中有多少个TransformerEncoderLayer堆叠在一起。
        '''
        # 两次Transfomer
        self.transfomer_encoder1 = TransformerEncoder(encoder_layer1,num_layers=2)
        self.transfomer_encoder2 = TransformerEncoder(encoder_layer2,num_layers=2)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        g_atts = self.tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transfomer_encoder2(src2)
        return memory
    
    def forward(self, src):
        l_atts = self.tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transfomer_encoder1(src1)
        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0))
        z2 = self.fcn(self.callback(src, x1))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  # (Batch, 1, output_channel)
        
