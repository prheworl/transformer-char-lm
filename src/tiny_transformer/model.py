import math, torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(T, device):
    m = torch.full((1,1,T,T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x, start=0):
        T = x.size(1)
        return x + self.pe[start:start+T, :]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dk = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.h,self.dk).transpose(1,2)
        k = k.view(B,T,self.h,self.dk).transpose(1,2)
        v = v.view(B,T,self.h,self.dk).transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.dk)
        if attn_mask is not None: attn = attn + attn_mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,T,D)
        return self.proj_drop(self.proj(out))

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop=0.1, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, drop, drop)
        self.ffn = PositionwiseFFN(d_model, d_ff, drop)
    def forward(self, x, attn_mask=None):
        if self.pre_ln:
            y = x + self.mha(self.ln1(x), attn_mask)
            z = y + self.ffn(self.ln2(y))
        else:
            y = self.ln1(x + self.mha(x, attn_mask))
            z = self.ln2(y + self.ffn(y))
        return z

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=4,
                 block_size=256, drop=0.1, pre_ln=True, use_pos=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_pos = use_pos
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=block_size+2048)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, drop, pre_ln)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)
        if self.use_pos: x = self.pos_enc(x)
        x = self.drop(x)
        mask = causal_mask(T, idx.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss