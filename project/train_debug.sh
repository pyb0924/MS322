#!/bin/bash

for i in 0 1 2 3; do
  python train.py \
    --batch-size 16 \
    --fold $i\
    --root runs/UNet11_binary\
    --lr 0.0001 \
    --n-epochs 10 \
    --type binary\
    --model UNet11

  python train.py \
    --batch-size 16 \
    --fold $i \
    --root runs/UNet11_binary\
    --lr 0.00001 \
    --n-epochs 20 \
    --type binary\
    --model UNet11\
    --ends-flag 1
done
