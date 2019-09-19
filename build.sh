#!/bin/bash

echo before

which python

export PATH="/home/public/miniconda3/bin:$PATH"

which python

python -c 'import sys; print(sys.version)'

python trainer.py

echo after
