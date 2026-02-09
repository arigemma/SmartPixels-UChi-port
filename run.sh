#!/bin/bash

for conf in 17180303/barrel-flat-pt/50x10 17180303/barrel-flat-pt/50x20 
do
    echo "Running config: $conf"
    python train.py -c config/$conf.yml --balance-classes -m towards-model-3 --device cuda -o results/trained_models/$conf/towards-model-3
    python train.py -c config/$conf.yml --balance-classes -m towards-model-2 --device cuda -o results/trained_models/$conf/towards-model-2
    python train.py -c config/$conf.yml --balance-classes -m towards-model-1 --device cuda -o results/trained_models/$conf/towards-model-1
done
