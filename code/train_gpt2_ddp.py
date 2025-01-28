#!python3
# vim: et:ts=4:sts=4:sw=4

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import inspect
import os
import numpy as np
# ---------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Regularisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # Batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()
        # Calculate query, key, values for all heads in batch and move head forward to be the batch size
        # nh is "number of heads", hs is "head)size", and C (number of channels( = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh * hs = C = 768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # The above six lines are replaed with the following to ensure
        # torchcompile applies flash attension
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:

    # Max sequence length
    block_size: int = 1024
    # Number of tokens: 50 000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257
    # Number of layers
    n_layer: int = 12
    # Number of heads
    n_head: int = 12
    # Embedding dimension
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # Forward the token and position embeddings
        # shape (T)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # Position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # Token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # Forward the blocks of teh transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, rank):
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"Sample {i}: {decoded}")

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # AdamW docs: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
        optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = self(x, y)
        loss = loss / grad_accum_steps
        return loss

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def get_shards(split):
    data_root = "/bask/projects/v/vjgo8416-training25/edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    return shards

class DataIterator:

    def __init__(self, B, T, num_processes, process_rank):
        self.B = B
        self.T = T
        self.num_processes = num_processes
        self.shards = get_shards("train")
        assert len(self.shards) > 0, f"No shards found"

        self.reset(process_rank)

    def reset(self, process_rank):
        self.process_rank = process_rank
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        assert (len(self) * self.B * self.T * self.num_processes) < (len(self.tokens) * len(self.shards)), f"Not enough data for a complete epoch"

    def __len__(self):
        return total_tokens // (self.B * self.T * self.num_processes)

    def __iter__(self):
        B, T = self.B, self.T
        for _ in range(len(self)):
            buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
            # Inputs
            x = (buf[:-1]).view(B, T)
            # Targets
            y = (buf[1:]).view(B, T)
            # Advance the position in the tensor
            self.current_position += B * T * self.num_processes
            # If loading the next batch would be out of bounds, advance to next shard
            if self.current_position + (B * T * self.num_processes) + 1 > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
            yield x, y

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_tokens = 9799991296
minibatch_size = 524288
B = 32
T = 1024
grad_accum_steps = minibatch_size // (B * T * world_size)
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = total_tokens // minibatch_size

assert minibatch_size % (B * T * world_size) == 0, "Make sure minibatch size is divisible by microbatch size"

config = GPTConfig(vocab_size=50304)
train_dataset = DataIterator(B=B, T=T, num_processes=world_size, process_rank=ddp_rank)
train_loader = iter(train_dataset)

torch.set_float32_matmul_precision("high")

raw_model = GPT(config)

raw_model.to(device)
model = DDP(raw_model, device_ids=[ddp_local_rank])

optimizer = raw_model.configure_optimizers()

for step in range(max_steps):
    if step % 25 == 0:
        model.eval()
        raw_model.generate(ddp_rank)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        batch = next(train_loader)
        raw_model.require_backward_grad_sync = (micro_step == (grad_accum_steps - 1))
        loss = raw_model.training_step(batch, (step * grad_accum_steps) + micro_step)
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    print(f"| step {step:4d}/{max_steps} | loss {loss_accum.item():0.6f} | lr: {lr:0.4e} | norm: {norm:0.4f} |")

model.eval()

destroy_process_group()
