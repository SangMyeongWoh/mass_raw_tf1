#!/bin/sh

python3 main.py --op finetune \
                --workspace EX18_multi_turn \
                --preprocess_workspace EX2_multi_turn \
                --param_set et_big_finetune \
                --model_type et \
                --target_gpu 1 \
                --batch_size 16 \
                --checkpoint_type p \
                --ckp_workspace EX16_max_pe \
                --init_checkpoint model-450000 \
                --seq_len_encoder 256 \
                --seq_len_decoder 64 \
                --is_multi_turn