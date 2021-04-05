#!/bin/sh

python3 main.py --op api \
                --batch_size 1 \
                --param_set et_big_finetune \
                --preprocess_workspace EX2_multi_turn \
                --target_gpu -1 \
                --checkpoint_type f \
                --ckp_workspace EX18_multi_turn \
                --init_checkpoint model-2450000 \
                --model_type et \
                --seq_len_encoder 256 \
                --seq_len_decoder 64 \
                --is_multi_turn \
                --port 3043