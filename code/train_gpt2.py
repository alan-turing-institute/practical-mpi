#!python3
# vim: et:ts=4:sts=4:sw=4

import lightning as L
from torch.utils.data import IterableDataset, Dataset, DataLoader
from lightning import Trainer

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

class GPT(L.LightningModule):

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
        xgen = self.transfer_batch_to_device(tokens, self.device, 0)
        sample_rng = torch.Generator(device=self.device)
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
        # OneCycleLR docs: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=1, steps_per_epoch=max_steps, pct_start=(warmup_steps / max_steps), anneal_strategy="cos", cycle_momentum=True, base_momentum=0.0, max_momentum=0.0, div_factor=warmup_steps, final_div_factor=(max_lr / min_lr))
        return { "optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        loss = loss / grad_accum_steps
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % (200 * world_size) == 0:
            self.generate(self.trainer.global_rank)

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

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    global_id = (torch.distributed.get_rank() * torch.distributed.get_world_size()) + worker_info.id
    worker_info.dataset.reset(global_id)

class DataIterator(IterableDataset):

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

nnodes = int(os.environ.get("SLURM_NNODES", 1))
world_size = nnodes * torch.cuda.device_count()

L.pytorch.seed_everything(1337, workers=True)

enc = tiktoken.get_encoding("gpt2")

total_tokens = 9799991296
minibatch_size = 524288
B = 16
T = 1024
grad_accum_steps = minibatch_size // (B * T * world_size)
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = total_tokens // (B * T * world_size)

assert minibatch_size % (B * T * world_size) == 0, "Make sure minibatch size is divisible by microbatch size"

config = GPTConfig(vocab_size=50304)
train_dataset = DataIterator(B=B, T=T, num_processes=world_size, process_rank=0)
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=1, worker_init_fn=worker_init_fn)

torch.set_float32_matmul_precision("high")

model = GPT(config)

trainer = Trainer(
    max_epochs=1,
    gradient_clip_val=1.0,
    num_nodes=nnodes,
    accelerator="gpu",
    strategy="ddp",
)
trainer.fit(model, train_loader)

