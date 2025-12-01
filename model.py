from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken
import time

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    block_size: int = 1024
    n_embed: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        #key, query, value projections for all heads, one projection for all heads
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        #output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head size)
        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, n_head, T, head size)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = nn.functional.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "wpe": nn.Embedding(config.block_size, config.n_embed),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embed),
        })
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(idx)
        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained model weights from Hugging Face."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print(f"Loading weights from pretrained model: {model_type}")
        config_args = {
            "gpt2": dict    (n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=24, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]
        sd_keys = sorted(sd_keys)


        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = sorted(sd_keys_hf)
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert sd_keys_hf == sd_keys, f"Mismatch in number of keys: {len(sd_keys_hf)} != {len(sd_keys)}"


        for k in sd_keys_hf:
            if any(k.endswith(suffix) for suffix in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

num_samples = 5
max_seq_len = 50

model = GPT.from_pretrained("gpt2")
model.eval()
model.to("cuda")

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode("Hello, I'm a language model")
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1)
x = x.to("cuda")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_seq_len:
    logits = model(x)
    logits = logits[:, -1, :] 
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    x_i = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
    x_next = torch.gather(topk_indices, dim=-1, index=x_i) # (B, 1)
    x = torch.cat((x, x_next), dim=1)

for i in range(num_samples):
    print(tokenizer.decode(x[i].tolist()))
