from absl import app
from absl import flags


flag_objs = flags.FLAGS


def define_flags():
  # parameter - common
  flags.DEFINE_string("op", None, "operator")
  flags.DEFINE_string("workspace", "0921_tyr_newdata_fromelastic", "")
  flags.DEFINE_integer("seq_len_encoder", 128, "")
  flags.DEFINE_integer("seq_len_decoder", 64, "Only for fine-tuning")
  flags.DEFINE_boolean("is_ner", False, "")
  flags.DEFINE_boolean("is_multi_turn", False, "for finetune")

  # parameter - tokenizer
  flags.DEFINE_string("tokenizer_name", "2020-06-18_wiki_30000.model", "")

  # parameter - preprocess
  flags.DEFINE_string("pretrain_data_dir", "sentencepiece_tyr", "")
  flags.DEFINE_integer("num_process", 9, "The number of multiprocessing unit")
  flags.DEFINE_integer("num_rand_mask", 5, "The number of dataset for dynamic masking")
  flags.DEFINE_float("mask_prob", 0.15, "for pretrain")
  flags.DEFINE_integer("block_size", 32, "for pretrain")
  flags.DEFINE_integer("min_seq_len", 5, "for pretrain & finetune")
  flags.DEFINE_boolean("do_conversation", False, "Preprocess conversation data for pretrain")
  flags.DEFINE_integer("max_multi_turn", 7, "for finetune")

  # parameter - train
  flags.DEFINE_string("preprocess_workspace", "EX1_single_turn", "Preprocessed dataset for training")
  flags.DEFINE_string("param_set", "base_pretrain", "See ./configs/model_params.py")
  flags.DEFINE_string("model_type", "et", "vanilla transformer (vt), evolved transformer (et), ner transformer (nt)")
  flags.DEFINE_integer("target_gpu", 0, "")
  flags.DEFINE_float("gpu_usage", 1.0, "")
  flags.DEFINE_integer("batch_size", 16, "")
  flags.DEFINE_string("ckp_workspace", "EX1", "Workspace for restoring checkpoint")
  flags.DEFINE_string("init_checkpoint", None, "")
  flags.DEFINE_string("checkpoint_type", "p", "Two options: 'p' - pretrain, 'f' - finetune")

  # parameter - api
  flags.DEFINE_string("host", "0.0.0.0", "")
  flags.DEFINE_integer("port", 4999, "")
  flags.DEFINE_boolean("do_sample", False, "top-k sampling if true else greedy search")
  flags.DEFINE_integer("top_k", 10, "")

def main(argv):
  del argv

  if flag_objs.op == 'build_tokenizer' or flag_objs.op == 'bt':
    from src.build_tokenizer import BuildTokenizerTask
    task = BuildTokenizerTask(flag_objs)
    task.fit()

  if flag_objs.op == 'preprocess_conversation' or flag_objs.op == 'pc':
    from src.utterance_handler import PreprocessUtteranceTask
    task = PreprocessUtteranceTask(flag_objs)
    task.run(do_all=False)

  if flag_objs.op == 'translate' or flag_objs.op == 'tl':
    from src.translator import TranslateTask
    task = TranslateTask(flag_objs)
    task.run()

  if flag_objs.op == 'apply_ner' or flag_objs.op == 'an':
    from src.ner_handler import NERHandler
    task = NERHandler(flag_objs)
    task.apply_ner(file_names=['raw_nia', 'data', 'raw_flagship', 'blend', 'conv', 'empathy', 'wizard'])

  if flag_objs.op == 'generate_pretrain' or flag_objs.op == 'gpt':
    from src.data_handler import HandleDatasetTask
    task = HandleDatasetTask(flag_objs, mode='pretrain')
    if not flag_objs.do_conversation:
      task.create_dataset()
    else:
      task.create_dataset(file_names=['raw_flagship', 'raw_nia', 'data', 'conv', 'empathy', 'wizard'])

  if flag_objs.op == 'generate_finetune' or flag_objs.op == 'gft':
    from src.data_handler import HandleDatasetTask
    task = HandleDatasetTask(flag_objs, mode='finetune')
    task.create_dataset(file_names=['raw_flagship', 'raw_nia', 'data', 'conv', 'empathy', 'wizard', 'blend'])
    #task.create_dataset(file_names=['raw_flagship', 'raw_nia', 'data'])


  if flag_objs.op == 'pretrain' or flag_objs.op == 'pt':
    from src.amumal_model import AMUMAL
    task = AMUMAL(flag_objs, is_training=True, mode='p')
    task.fit(mode='pretrain')

  if flag_objs.op == 'finetune' or flag_objs.op == 'ft':
    from src.amumal_model import AMUMAL
    task = AMUMAL(flag_objs, is_training=True, mode='f')
    task.fit(mode='finetune')

  if flag_objs.op == 'finetune' or flag_objs.op == 'ft':
    from src.amumal_model import AMUMAL
    task = AMUMAL(flag_objs, is_training=True, mode='f')
    task.fit(mode='finetune')

  if flag_objs.op == 'api' or flag_objs.op == 'a':
    from src.api import run_api
    run_api()


if __name__ == '__main__':
  define_flags()
  app.run(main)
