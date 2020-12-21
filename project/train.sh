#!/bin/bash

for model in 'UNet' 'UNet11' 'UNet16' 'LinkNet34'; do
  for i in 0 1 2 3; do
    for type in 'parts' 'instruments'; do
      python train.py \
        --batch-size 16 \
        --fold $i \
        --root runs/${model}_${type} \
        --lr 0.0001 \
        --n-epochs 10 \
        --type $type \
        --model $model

      python train.py \
        --batch-size 16 \
        --fold $i \
        --root runs/${model}_${type} \
        --lr 0.00001 \
        --n-epochs 20 \
        --type $type \
        --model $model \
        --ends-flag 1
    done
  done
done
