#!/bin/bash

echo before

export PATH="/home/yegor/miniconda3/bin:$PATH"

python trainer.py

echo after
