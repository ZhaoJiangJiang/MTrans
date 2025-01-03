import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert embed_size % self.num_heads == 0, "Embedding size must be divisible by number of heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query)       # (batch_size, seq_len, embed_size)
        K = self.key(key)           # (batch_size, seq_len, embed_size)
        V = self.value(value)       # (batch_size, seq_len, embed_size)

        # 拆分为多个头
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算 Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, float('-1e20'))  # 应用mask

        attention = torch.softmax(energy, dim=-1)   # (batch_size, num_heads, seq_len, seq_len)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)    # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头的输出
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_len, embed_size)

        out = self.fc_out(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hid_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, embed_size)
        )

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention_out = self.multi_head_attention(query, key, value, mask)
        # 残差连接 + 层归一化
        x = self.layer_norm1(attention_out + query)

        # 前馈网络
        ff_out = self.fc(x)
        # 残差连接 + 层归一化
        out = self.layer_norm2(ff_out + x)
        return out


class MTransModel(nn.Module):
    def __init__(self, embed_size=8, num_heads=4, num_layers=2, ff_hid_dim=64, num_classes=10, dropout=0.1):
        super(MTransModel, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.positional_encoding = nn.Parameter(torch.randn(8, embed_size))

        # Transformer Encoder 层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, num_heads, ff_hid_dim, dropout)
            for _ in range(num_layers)
        ])

        # 最后的分类全连接层
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=8, feature_size=8)
        seq_len = x.size(1)

        # # 添加位置编码
        # x = x + self.positional_encoding[:seq_len, :]

        # 通过 Transformer 编码器层
        for layer in self.encoder_layers:
            x = layer(x, x, x)  # 自注意力通常输入的查询、键、值都是同一个输入

        # 使用最后的输出进行分类
        x = x.mean(dim=1)  # 池化操作，求平均
        out = self.fc_out(x)  # 分类输出
        return out
