#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

python trainer.py


# by this point, trainer.py should have added whatever to this directory

git rm --cached .
git add *.h5
git commit -a -m "auto-push: $BUILD_NUMBER"
