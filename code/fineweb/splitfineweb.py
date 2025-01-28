#!python3
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright © 2024 David Llewellyn-Jones

import os
import torch
import numpy as np

local_dir = "edu_fineweb10B"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

def load_tokens(filename):
    print(f"Loading file: {filename}")
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def write_datafile(filename, tokens_np):
    print(f"Writing datafile: {filename}")
    np.save(filename, tokens_np)
    print(f"File written: {filename}")

filename = os.path.join(DATA_CACHE_DIR, f"input.npy")
ptt = load_tokens(filename)

number = 10
size = len(ptt) // number
print(f"Total count: {number}")
print(f"Length per file: {size}")

for count in range(number):
    buf = ptt[size * count : size * (count + 1)]

    shard_index = count + 1
    filename = os.path.join(DATA_CACHE_DIR, f"fineweb_train_{shard_index:06d}.npy")
    write_datafile(filename, buf)

