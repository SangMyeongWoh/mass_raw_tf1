#!/bin/sh

python3 main.py --op pretrain \
                --workspace EX10_no_bos \
                --preprocess_workspace EX1 \
                --param_set et_big_pretrain \
                --model_type et \
                --target_gpu 0 \
                --batch_size 32