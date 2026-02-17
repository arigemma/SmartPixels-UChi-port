#!/bin/bash

# for conf in 17180303/barrel-flat-pt/50x10 17180303/barrel-flat-pt/50x20 
# do
echo "Running config: v3"
python train.py -c config/v3.yml --balance-classes -m towards-model-3 --device cuda -o results/trained_models/towards-model-3 --epochs 5 -bs 1024 -lr 0.001
python train.py -c config/v3.yml --balance-classes -m towards-model-2 --device cuda -o results/trained_models/towards-model-2 --epochs 5 -bs 1024 -lr 0.001
python train.py -c config/v3.yml --balance-classes -m towards-model-1 --device cuda -o results/trained_models/towards-model-1 --epochs 5 -bs 1024 -lr 0.001
# done
