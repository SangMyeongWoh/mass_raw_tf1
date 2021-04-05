#!/bin/sh

python3 main.py --op api \
                --batch_size 1 \
                --param_set et_big_finetune \
                --target_gpu -1 \
                --checkpoint_type f \
                --ckp_workspace EX12_single_turn \
                --init_checkpoint model-550000 \
                --model_type et \
                --port 3041