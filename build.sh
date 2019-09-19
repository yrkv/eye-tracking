#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

python trainer.py

# so that it doesn't yell at me
#git config user.email "yegor@tydbits.com"
#git config user.name  "Yegor Kuznetsov"

# by this point, trainer.py should have added whatever to this directory

