#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

python trainer.py

# by this point, trainer.py should have added whatever to this directory


cp *.h5 $JENKINS_HOME/eye-tracking-models
git checkout master
git status
git add *.h5
git commit -a -m "auto-push: $BUILD_NUMBER"
git remote add origin https://github.com/yrkv/eye-tracking-models.git
git push -u origin master
