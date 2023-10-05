
#! /usr/bin/env bash

# You can use this script to ssh into a GPU node.

# Arch must be one of:
# @@rtx
# @@dgx
# @@2080
# @@v100

arch=@@rtx
ngpu=1

qrsh -q gpu.q${arch} -now no -cwd -l num_proc=10,mem_free=12G,h_rt=16:00:00,gpu=$ngpu

# eof