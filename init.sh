#!/usr/bin/bash

if [ ! -d 'ven' ]; then

    virtualenv ven
fi
source ven/bin/activate
pip3 install -r requirements.txt
