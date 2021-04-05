#!/bin/sh

python3 main.py --op finetune \
                --workspace EX19_single_ner \
                --preprocess_workspace EX3_single_ner \
                --param_set et_big_finetune \
                --model_type nt \
                --target_gpu 1 \
                --batch_size 16 \
                --checkpoint_type p \
                --ckp_workspace EX11_light \
                --init_checkpoint model-500000 \
                --seq_len_encoder 64 \
                --seq_len_decoder 64 \
                --is_ner