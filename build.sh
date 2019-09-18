#!/bin/bash

echo before

exec /home/yegor/miniconda3/bin/python -c 'import sys; print(sys.version)'

python trainer.py

echo after
