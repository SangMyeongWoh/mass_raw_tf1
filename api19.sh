#!/bin/sh

python3 main.py --op api \
                --batch_size 1 \
                --param_set et_big_finetune \
                --preprocess_workspace EX3_single_ner \
                --target_gpu -1 \
                --checkpoint_type f \
                --ckp_workspace EX19_single_ner \
                --init_checkpoint model-950000 \
                --model_type nt \
                --seq_len_encoder 64 \
                --seq_len_decoder 64 \
                --is_multi_turn \
                --port 3043 \
                --is_ner