#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

mkdir -p data/test
mkdir -p data/train

python trainer.py
