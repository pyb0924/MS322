#!/bin/bash

for model in 'UNet' 'UNet11' 'UNet16' 'AlbuNet' 'LinkNet34';do
    for i in 0 1 2 3; do
      python train.py \
        --batch-size 16 \
        --fold $i \
        --root runs2/${model}_instruments\
        --lr 0.0001 \
        --n-epochs 8 \
        --type instruments\
        --model $model

      python train.py \
        --batch-size 16 \
        --fold $i \
        --root runs2/${model}_instruments\
        --lr 0.00001 \
        --n-epochs 16 \
        --type instruments\
        --model $model\

      python train.py \
        --batch-size 16 \
        --fold $i \
        --root runs2/${model}_instruments\
        --lr 0.000001 \
        --n-epochs 24 \
        --type instruments\
        --model $model\
        --ends-flag 1
    done
done
