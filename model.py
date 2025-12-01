from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken
import time
import sys

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
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        
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
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

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
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std = 1/math.sqrt(2*self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(idx)
        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
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

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open("test_set.txt", "r") as f:
            text = f.read()
        tokens = tokenizer.encode(text) # (n_tokens,)
        self.tokens = torch.tensor(tokens)
        self.current_idx = 0
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"Number of batches per epoch: {len(self.tokens) // (B * T)}")

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_idx:self.current_idx+B*T+1] # (B*T+1,)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_idx += B*T
        if self.current_idx > len(self.tokens) - B*T - 1:
            self.current_idx = 0
        return x, y

device ="cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)
elif device == "mps":
    torch.mps.manual_seed(42)


tokenizer = tiktoken.get_encoding("gpt2")

train_loader = DataLoader(B=8, T=1024)
#model = GPT.from_pretrained("gpt2")k
model = GPT(GPTConfig())
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for iter in range(50):
    time_start = time.time()
    x, y = train_loader.get_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    time_elapsed = time.time()-time_start
    print(f"Step {iter}, Loss: {loss.item()}, Time: {time_elapsed*1000:.2f}ms")


model.eval()
test_text = "Hello, how are you?"
test_tokens = tokenizer.encode(test_text)
num_samples = 4
test_tokens = torch.tensor(test_tokens, dtype=torch.long, device=device).unsqueeze(0)
test_tokens = test_tokens.repeat(num_samples, 1)
print(test_tokens.shape)
with torch.no_grad():
    
    while test_tokens.size(1) < 100:
        logits, _ = model(test_tokens)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        test_tokens = torch.cat((test_tokens, next_token), dim=1)
for i in range(num_samples):
    print(tokenizer.decode(test_tokens[i].tolist()))
    print("-"*100)