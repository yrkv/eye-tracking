#!/bin/bash

echo before

which python

export PATH="/home/yegor/miniconda3/bin:$PATH"

which python

python -c 'import sys; print(sys.version)'

python trainer.py

echo after
