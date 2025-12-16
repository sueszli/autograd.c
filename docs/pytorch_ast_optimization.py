import ast
import astunparse

import torch
from torch import nn
import torch.nn.functional as F

import os
import time
from dataclasses import dataclass


# source: https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
csa_str = """
import torch
import math
from torch import nn
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        ###!!! Beginning of the scaled dot product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ###!!! End of the scaled dot product attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
"""


def get_exectime(python_str: str) -> float:
    exec(python_str, globals())
    assert "CausalSelfAttention" in globals(), "class not found in namespace"

    @dataclass
    class Config:
        n_embd: int = 3072  # embedding dimensionality
        n_head: int = 16  # number of attention heads
        dropout: float = 0.1  # dropout probability
        bias: bool = True  # whether to use bias in projection layers

    def gen_bias(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, -1e10).unsqueeze(0).unsqueeze(1)
        return mask

    config = Config()
    model = CausalSelfAttention(config)  # type: ignore
    setattr(model, "bias", gen_bias(config.n_embd))  # missing attribute in the original code
    x = torch.rand(16, 1024, 3072, dtype=torch.float32)

    t_start = time.time()
    output = model.forward(x)
    t_end = time.time()
    return t_end - t_start


def transform(python_str: str) -> str:
    tree = ast.parse(python_str)
    forward_method = tree.body[3].body[1]  # type: ignore
    forward_method.body[5:10] = ast.parse("y = F.scaled_dot_product_attention(q, k, v)").body
    return astunparse.unparse(tree)


def main():
    os.system("clear" if os.name == "posix" else "cls")

    new_csa_str = transform(csa_str)
    print(f"Original code:\n\n{csa_str}\n\nOptimized code:\n\n{new_csa_str}\n\n")

    iters = 5
    print(f"Running {iters} benchmark iterations...")

    avg_t1 = 0.0
    for i in range(iters):
        score = get_exectime(csa_str)
        print(f"\t{i + 1}/{iters} - Original time: {score}")
        avg_t1 += score
    avg_t1 /= iters

    avg_t2 = 0.0
    for i in range(iters):
        score = get_exectime(new_csa_str)
        print(f"\t{i + 1}/{iters} - Optimized execution time: {score}")
        avg_t2 += score
    avg_t2 /= iters

    print(f"Avg original time: {avg_t1}s")  # Avg original time: 2.4412187576293944s
    print(f"Avg optimized execution time: {avg_t2}s")  # Avg optimized execution time: 0.9933661937713623s
    print(f"🔥 Avg speedup: {avg_t1 / avg_t2 * 100:.2f}%")  # 🔥 Avg speedup: 245.75%
    print(f"🔥 Avg seconds saved: {avg_t1 - avg_t2:.2f}s")  # 🔥 Avg seconds saved: 1.45s


if __name__ == "__main__":
    main()
