#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

python trainer.py


cd /home/public/eye-tracking-models

git add .
git commit -a -m "$BUILD_NUMBER"
git push

