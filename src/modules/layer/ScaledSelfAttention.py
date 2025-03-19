import torch
from torch import nn

from modules.layer.PESymetry import PESymetryMean

class PESelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim=None, num_heads=1):
        super(PESelfAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim if embed_dim is not None else input_dim * num_heads
        self.num_heads = num_heads

        self.embed_q = PESymetryMean(self.input_dim, self.embed_dim)
        self.embed_k = PESymetryMean(self.input_dim, self.embed_dim)
        self.embed_v = PESymetryMean(self.input_dim, self.embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        q = nn.functional.elu(self.embed_q(x))
        k = nn.functional.elu(self.embed_k(x))
        v = nn.functional.elu(self.embed_v(x))

        att, _ = self.attention(q, k, v)
        # Output shape: [batch_size, indiv_nb, embed_dim
        return att



class VanillaScaledSelfAttention(nn.Module):
    def __init__(self, emb_dim, q_dim=None, v_dim=None):
        super(VanillaScaledSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.q_dim = q_dim if q_dim is not None else self.emb_dim * 10
        self.v_dim = v_dim if v_dim is not None else self.q_dim
        self.w_query = nn.Linear(self.emb_dim, self.q_dim)
        # Remark: q_dim == k_dim by definition
        self.w_key = nn.Linear(self.emb_dim, self.q_dim)
        self.w_value = nn.Linear(self.emb_dim, self.v_dim)
        # self.softmax = nn.Softmax(dim=-2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        # return F.scaled_dot_product_attention(query, key, value)
        qk = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)
        # qk = nn.ELU()(qk)
        attention = self.softmax(qk)
        output = torch.matmul(attention, value)
        return output


class ScaledSelfAttention(nn.Module):
    def __init__(self, emb_dim, q_dim=None, v_dim=None, bias=True):
        super(ScaledSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.q_dim = q_dim if q_dim is not None else self.emb_dim * 10
        self.v_dim = v_dim if v_dim is not None else self.q_dim
        self.w_query = nn.Linear(self.emb_dim, self.q_dim, bias=bias)
        # Remark: q_dim == k_dim by definition
        self.w_key = nn.Linear(self.emb_dim, self.q_dim, bias=bias)
        self.w_value = nn.Linear(self.emb_dim, self.v_dim, bias=bias)
        # self.softmax = nn.Softmax(dim=-2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.w_query(x)
        query = nn.Sigmoid()(query)
        key = self.w_key(x)
        value = self.w_value(x)
        # return F.scaled_dot_product_attention(query, key, value)
        qk = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)
        qk = nn.ELU()(qk)
        attention = self.softmax(qk)
        output = torch.matmul(attention, value)
        return output

class ScaledCrossAttention(nn.Module):
    def __init__(self, q_emb_dim, v_emb_dim, q_dim=None, v_dim=None, bias=True):
        super(ScaledCrossAttention, self).__init__()
        self.q_emb_dim = q_emb_dim
        self.v_emb_dim = v_emb_dim
        self.q_dim = q_dim if q_dim is not None else self.q_emb_dim * 10
        self.v_dim = v_dim if v_dim is not None else self.q_dim
        self.w_query = nn.Linear(self.q_emb_dim, self.q_dim, bias=bias)
        # Remark: q_dim == k_dim by definition
        self.w_key = nn.Linear(self.v_emb_dim, self.q_dim, bias=bias)
        self.w_value = nn.Linear(self.v_emb_dim, self.v_dim, bias=bias)
        # self.softmax = nn.Softmax(dim=-2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, x):
        query = self.w_query(q)
        query = nn.Sigmoid()(query)
        key = self.w_key(x)
        value = self.w_value(x)
        # return F.scaled_dot_product_attention(query, key, value)
        qk = torch.matmul(query, key.transpose(-2, -1)) / (self.q_emb_dim ** 0.5)
        qk = nn.ELU()(qk)
        attention = self.softmax(qk)
        output = torch.matmul(attention, value)
        return output

