#!python3
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright © 2024 David Llewellyn-Jones

import os
import torch
import numpy as np

data_root = "edu_fineweb10B"

def load_tokens(filename):
    print(f"Loading file: {filename}")
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def get_shards(split):
    data_root = "edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    return shards

shards = get_shards("train")
print(f"Shards: {len(shards)}")

total = 0
shard_index = 0
for shard in shards:
    ptt = load_tokens(shard)
    print(f"Shard {shard_index} tokens: {len(ptt)}")
    total += len(ptt)
    shard_index += 1

print(f"Total tokens: {total}")

