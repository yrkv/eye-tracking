#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

python trainer.py

# by this point, trainer.py should have added whatever to this directory

cp *.h5 $JENKINS_HOME/eye-tracking-models
git add .
git commit -a -m "auto-push: $BUILD_NUMBER"
git push
