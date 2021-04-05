#!/bin/sh

python3 main.py --op api \
                --batch_size 1 \
                --param_set base_finetune \
                --target_gpu -1 \
                --checkpoint_type f \
                --ckp_workspace EX9_more_data \
                --init_checkpoint model-400000 \
                --model_type vt \
                --port 3040