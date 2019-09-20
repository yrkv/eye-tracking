#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

python trainer.py

# by this point, trainer.py should have added whatever to this directory
# copy files to intermediate location, other project sends them off to repo
cp *.h5 $JENKINS_HOME/eye-tracking-models
