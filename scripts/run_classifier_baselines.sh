#!/bin/bash

for classifier in "RF" "MLP" "SVM" "regression"
do
    for zero_class in "remove" "keep"
    do
            python3 classifier_baselines.py \
                --classifier $classifier \
                --dataset train \
                --zero_class $zero_class
    done
done