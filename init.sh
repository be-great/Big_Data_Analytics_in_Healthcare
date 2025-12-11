#!/usr/bin/bash
# ------------------------
# Purpose: create/activate Python virtual environment and install dependencies.
# Objects:
#   - "ven": folder representing the virtual environment.
#   - virtualenv: creates the environment if missing.
#   - activate script: switches shell into the environment.
#   - pip installer: installs packages from requirements.txt.
# Flow:
#   1) Check "ven" folder.
#   2) Create if missing.
#   3) Activate environment.
#   4) Install requirements.
# -----------------------


if [ ! -d 'ven' ]; then

    virtualenv ven
fi
source ven/bin/activate
pip3 install -r requirements.txt
