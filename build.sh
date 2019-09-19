#!/bin/bash

export PATH="/home/public/miniconda3/bin:$PATH"

eval "$(ssh-agent -s)"

ssh-add $JENKINS_HOME/.ssh/id_sa

git clone git@github.com:yrkv/eye-tracking-models.git $JENKINS_HOME/eye-tracking-models

python trainer.py

cd $JENKINS_HOME/eye-tracking-models

# so that it doesn't yell at me
git config user.email "yegor@tydbits.com"
git config user.name  "Yegor Kuznetsov"

# by this point, trainer.py should have added whatever to this directory
git add .
git commit -a -m "$BUILD_NUMBER"
git push

