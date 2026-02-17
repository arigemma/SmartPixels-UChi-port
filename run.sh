#!/bin/bash

# for conf in 17180303/barrel-flat-pt/50x10 17180303/barrel-flat-pt/50x20 
out_dir=results/2026_02_16
for conf in 10783560/v3
do
    echo "Running config: $conf"
    python train.py -c config/$conf.yml --balance-classes -m towards-model-3 --device cuda -o $out_dir/$conf/towards-model-3 -bs 1024
    python train.py -c config/$conf.yml --balance-classes -m towards-model-2 --device cuda -o $out_dir/$conf/towards-model-2 -bs 1024
    python train.py -c config/$conf.yml --balance-classes -m towards-model-1 --device cuda -o $out_dir/$conf/towards-model-1 -bs 1024
    # python train.py -c config/$conf.yml --balance-classes -m lgn-dense-1 --device cuda -o $out_dir/$conf/lgn-dense-1 -bs 1024 -lr 0.01 --compile-model
    # python train.py -c config/$conf.yml --balance-classes -m lgn-dense-2 --device cuda -o $out_dir/$conf/lgn-dense-2 -bs 1024 -lr 0.01 --compile-model
    # python train.py -c config/$conf.yml --balance-classes -m lgn-conv-3 --device cuda -o $out_dir/$conf/lgn-conv-3 -bs 1024 -lr 0.01 --compile-model
done
