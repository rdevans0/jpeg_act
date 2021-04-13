#!/bin/bash

set -x
set -e


while read REQ; do 
    pip3 install $REQ; 
done < requirements.txt

set +x
set +e
