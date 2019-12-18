#!/bin/bash

# complete has 194 datasets
raw_data_path=./data/complete_classification.tar.xz
test_size=44
test_split_seed=3746673648

python3 -m dna split-data \
    --data-path $raw_data_path \
    --test-size $test_size \
    --split-seed $test_split_seed
