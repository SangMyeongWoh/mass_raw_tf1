from copy import deepcopy

PRETRAIN_MODEL_PARAMS = dict(
  num_layers=6,
  d_model=768,
  dff=3072,
  num_heads=12,
  dropout_rate=0.,
  is_embed_pos=True,
  learning_rate=5e-5,
  num_train_steps=10000000,
  num_warmup_steps=10000,
  num_epochs=1000,
  is_finetune=False
)

FINETUNE_MODEL_PARAMS = deepcopy(PRETRAIN_MODEL_PARAMS)
FINETUNE_MODEL_PARAMS['is_finetune'] = True

PRETRAIN_EVOLVED_PARAMS = dict(
  num_encoder_layers=1,
  num_decoder_layers=13,
  d_model=512,
  dff=2048,
  num_heads=8,
  dropout_rate=0.,
  is_embed_pos=True,
  type_size=16,
  learning_rate=5e-5,
  num_train_steps=10000000,
  num_warmup_steps=10000,
  num_epochs=1000,
  is_finetune=False,
  use_lamb=False
)

FINETUNE_EVOLVED_PARAMS = deepcopy(PRETRAIN_EVOLVED_PARAMS)
FINETUNE_EVOLVED_PARAMS['is_finetune'] = True

LAMB_PARAMS = deepcopy(PRETRAIN_MODEL_PARAMS)
LAMB_PARAMS['learning_rate'] = 0.00176
LAMB_PARAMS['use_lamb'] = True
LAMB_PARAMS['num_warmup_steps'] = 100000

BIG_ET_PRETRAIN_PARAMS = deepcopy(PRETRAIN_EVOLVED_PARAMS)
BIG_ET_PRETRAIN_PARAMS['d_model'] = 1024
BIG_ET_PRETRAIN_PARAMS['dff'] = 4096
BIG_ET_PRETRAIN_PARAMS['num_heads'] = 16

BIG_ET_FINETUNE_PARAMS = deepcopy(BIG_ET_PRETRAIN_PARAMS)
BIG_ET_FINETUNE_PARAMS['is_finetune'] = True

PARAMS_MAP = dict(
  base_pretrain=PRETRAIN_MODEL_PARAMS,
  base_finetune=FINETUNE_MODEL_PARAMS,
  evolved_pretrain=PRETRAIN_EVOLVED_PARAMS,
  evolved_finetune=FINETUNE_EVOLVED_PARAMS,
  lamb_test=LAMB_PARAMS,
  et_big_pretrain=BIG_ET_PRETRAIN_PARAMS,
  et_big_finetune=BIG_ET_FINETUNE_PARAMS
)