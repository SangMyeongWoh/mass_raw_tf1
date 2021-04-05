AMUMAL(pretrain module by MASS + Evolved Transformer by MEENA)
------------------------------------------------
## Dependency
* python3, tensorflow1, numpy, absl, ...

## Directory
* `apis/` :
    - amumal.py: blueprints of flask
    - manager.py: generate key, memory conversation
* `configs/` : constant, model parameters, ner_info
* `model/` :
    - evolved_transformer.py: evolved transformer referenced by tensor2tensor
    - ner_transformer.py: transformer with ner result
    - transformer.py: main codes of transformer
    - utils.py: sub codes for transformer
* `src/` :
    - amumal_model.py: pretrain and finetune neural network, generate sentence
    - api.py: api for amumal model
    - build_tokenizer.py: train SentencePiece
    - data_handler.py: generate pretrain/finetune dataset
    - ner_handler.py: generate finetune dataset for ner transformer
    - translator.py: translate preprocessed conversation from English to Korean
    - utterance_handler.py: preprocess raw conversation in English
* `test/` : codes for testing
* `utils/` : sentencepiece wrapper, text filter, reader/writer for .txt files
* `data/` : raw data, data for pre-train and fine-tuning
    - wiki: for backup
    - clean: wiki and clien data with text filtering (for sentencepiece and pretrain)
    - utterance: raw conversation data in english
    - translate: preprocessed data of utterance
    - conversation: conversation data (for finetune)
    - ner: conversation data with ner result (for finetune)
    - sentencepiece: data to train SentencePiece model
    - pretrain: data to pre-train AMUMAL model
    - finetune: data to fine-tune AMUMAL model
* `checkpoints/` : trained model checkpoint
    - pretrain: workspaces for pre-trained model
    - finetune: workspaces for fine-tuned model
* `tokenizer/` : single token dictionary, trained sentencepiece model


## Build Tokenizer

Fit SentencePiece
```
python3 main.py --op build_tokenizer
```

## Preprocess Conversation in English for Translation

preprocess raw conversation in English (same shape with ./data/conversation)
```
python3 main.py --op preprocess_conversation
```

## Translation

translate preprocessed conversation from English to Korean
(check client_id & client_secret in './configs/constants.py')
```
python3 main.py --op translate
```

## Apply Named Entity Recognition

apply NER to conversation data
```
python3 main.py --op apply_ner
```

## Build Wiki & Clien Dataset for Pretraining

generate json and tfrecord dataset
```
python3 main.py --op generate_pretrain \
                --workspace EX1 \
                --pretrain_data_dir clean \
                --num_process 9 \
                --num_rand_mask 5 \
                --min_seq_len 5
```

## Build Conversation Dataset for Pretraining

generate json and tfrecord dataset
```
python3 main.py --op generate_pretrain \
                --workspace EX1 \
                --pretrain_data_dir conversation \
                --do_conversation \
                --seq_len_encoder 128
```

## Build single-turn Dataset for Finetuning

generate json and tfrecord dataset
```
python3 main.py --op generate_finetune \
                --workspace EX1_single_turn \
                --num_process 9 \
                --min_seq_len 5 \
                --seq_len_encoder 64 \
                --seq_len_decoder 64
```

## Build multi-turn Dataset for Finetuning

generate json and tfrecord dataset
```
python3 main.py --op generate_finetune \
                --workspace EX2_multi_turn \
                --is_multi_turn \
                --seq_len_encoder 256 \
                --seq_len_decoder 64
```

## Build single-turn Dataset for Finetuning with NER

generate json and tfrecord dataset
```
python3 main.py --op generate_finetune \
                --workspace EX3_single_ner \
                --seq_len_encoder 64 \
                --seq_len_decoder 64 \
                --is_ner
```

## Pretrain

Pretrain MASS
```
python3 main.py --op pretrain \
                --workspace [WORKSPACE] \
                --preprocess_workspace [WORKSPACE] \
                --param_set evolved_pretrain \
                --model_type et \
                --target_gpu 0 \
                --batch_size 64 \
                --seq_len_encoder 128
```

## Finetune

Pretrain MASS
```
python3 main.py --op finetune \
                --workspace [WORKSPACE] \
                --preprocess_workspace [WORKSPACE] \
                --param_set evolved_finetune \
                --model_type et \
                --target_gpu 0 \
                --batch_size 16 \
                --checkpoint_type p \
                --ckp_workspace [WORKSPACE] \
                --init_checkpoint model-[STEP NUMBER] \
                --seq_len_encoder 256 \
                --seq_len_decoder 64
```

## Finetune with NER

Pretrain MASS
```
python3 main.py --op finetune \
                --workspace [WORKSPACE] \
                --preprocess_workspace [WORKSPACE] \
                --param_set evolved_finetune \
                --model_type et \
                --target_gpu 0 \
                --batch_size 32 \
                --checkpoint_type p \
                --ckp_workspace [WORKSPACE] \
                --init_checkpoint model-[STEP NUMBER] \
                --seq_len_encoder 64 \
                --seq_len_decoder 64 \
                --is_ner
```

## Run API

run api of fintune model
```
python3 main.py --op api \
                --port 3040 \
                --nodo_sample \
                --preprocess_workspace [WORKSPACE] \
                --param_set base_finetune \
                --model_type vt \
                --target_gpu -1 \
                --batch_size 1 \
                --checkpoint_type f \
                --ckp_workspace [WORKSPACE] \
                --init_checkpoint model-[STEP NUMBER] \
                --seq_len_encoder 256 \
                --seq_len_decoder 64
```