#!/bin/bash

echo before

/bin/python -c 'import sys; print(sys.version)'

export PATH="/home/yegor/miniconda3/bin:$PATH"

python -c 'import sys; print(sys.version)'

python trainer.py

echo after
