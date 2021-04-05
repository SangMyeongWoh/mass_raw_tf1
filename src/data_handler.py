import os
import json
import random
import copy
from multiprocessing import Process
from datetime import datetime
import tensorflow as tf
import numpy as np

from utils.sentencepiece_wrapper import SentencePieceWrapper
from configs.constants import Constants
from configs.ner_cluster import NER
from src.ner_handler import get_ner


class HandleDatasetTask(object):
  """text 데이터를 json이나 tfrecord로 바꾸는 클래스"""

  def __init__(self, flag_objs, mode):
    """
    :param flag_objs:
    :param mode:
      pretrain> 'p', 'pretrain'
      finetune> 'f', 'finetune'
    """
    super(HandleDatasetTask, self).__init__()
    self.flag_objs = flag_objs
    self.verbose = True

    if mode not in ['p', 'pretrain', 'f', 'finetune']:
      raise ValueError('Wrong input for mode: %s (only "p" and "f" are allowed)' % mode)

    self.is_pretrain = mode in ['p', 'pretrain']
    self.is_ner = flag_objs.is_ner
    self.is_multi = flag_objs.is_multi_turn

    self.tokenizer = SentencePieceWrapper(
      model_dir=Constants.TOKENIZER_DIR,
      model_name=flag_objs.tokenizer_name
    )

    self.pad_id = self.tokenizer.get_pad_id()
    self.bos_id = self.tokenizer.get_bos_id()
    self.eos_id = self.tokenizer.get_eos_id()
    self.sep_id = self.tokenizer.get_sep_id()
    self.cls_id = self.tokenizer.get_cls_id()

    self.seq_len_encoder = flag_objs.seq_len_encoder
    masked_length = int(self.seq_len_encoder * flag_objs.mask_prob)
    self.seq_len_decoder = masked_length if self.is_pretrain else flag_objs.seq_len_decoder

    self.context_len = self.seq_len_encoder
    self.context_turn = 1
    if self.flag_objs.is_multi_turn:
      self.context_turn = self.flag_objs.max_multi_turn
      self.context_len -= self.context_turn * self.flag_objs.min_seq_len
    self.target_len = self.seq_len_encoder
    if not self.is_pretrain:
      token_len = [1, 2][self.is_ner]
      self.target_len = self.seq_len_decoder - token_len

  def transform_text_to_input_feature(self, a_text, context_list=None, ner_texts_list=None, do=False):
    """
    1. strip / encode / check length <_preprocess_text>
    2. masking / squeeze_context / adding_special_tokens <_to_*_input_feature>
    3. padding
    4. assertion
    :param a_text: transform 대상 text | 'utterance2'
    :param context_list: context 정보 (finetune인 경우에만 존재) | ['utterance1', 'utterance2', 'utterance1', ...]
    :param ner_texts_list: ner 정보 | [[('word', ner_id), ('word2', ner_id2), ...], ...]
    :param do: True면 None을 return하지 않고 끝까지 진행
    :return:
      enc_in: 공통
      dec_in: 공통
      dec_out: train 때 사용
      masked_pos: is_pretrain 때 사용
      len_dec_in: generate 때 사용
    """
    if not isinstance(context_list, list):
      context_list = [context_list]

    # call NER API to fill ner_texts_list
    if self.is_ner and not ner_texts_list:
      # [[('word', id), ('word2', id2)], None, [...], ...]
      ner_texts_list = [self.zip_ner(get_ner(x)[1:]) for x in context_list]

    # encode text to id
    a_ids = self._preprocess_text(a_text, self.target_len, do=do)
    context_ids_list = [self._preprocess_text(x, self.context_len, do=do) for x in context_list] if context_list else []
    ner_ids_list = [self._preprocess_ner(x) for x in ner_texts_list] if ner_texts_list else []
    if (a_ids is None) or (None in context_ids_list):
      return

    # id to input feature
    masked_pos, utt_type = None, None
    if self.is_pretrain:
      enc_in, dec_in, dec_out, masked_pos = self._to_pretrain_input_feature(a_ids)
    else:
      enc_in, dec_in, dec_out, utt_type = self._to_finetune_input_feature(a_ids, context_ids_list, ner_ids_list)

    # padding
    len_dec_in = len(dec_in)
    pad_len_enc = self.seq_len_encoder - len(enc_in)
    pad_len_dec = self.seq_len_decoder - len_dec_in
    enc_in += [self.pad_id] * pad_len_enc
    dec_in += [self.pad_id] * pad_len_dec
    dec_out += [self.pad_id] * pad_len_dec
    if self.is_pretrain:
      pad_len = self.seq_len_decoder - len(masked_pos)
      masked_pos += [self.pad_id] * pad_len
    if self.is_multi:
      utt_type += [self.pad_id] * pad_len_enc

    # assertion
    assert len(enc_in) == self.seq_len_encoder
    assert len(dec_in) == self.seq_len_decoder
    assert len(dec_out) == self.seq_len_decoder
    if self.is_pretrain:
      assert len(masked_pos) == self.seq_len_decoder
    if self.is_multi:
      assert len(utt_type) == self.seq_len_encoder

    return enc_in, dec_in, dec_out, masked_pos, utt_type, len_dec_in

  def generate_wrapper(self, workspace_dir, raw_data_file_name, cnt, pidx, data):
    """
    1. setting TrainWriter
    2. extract (context, utterance, ner_info) from data
    3. transform <transform_text_to_input_feature>
    4. to example <_create_example>
    5. write json & tfrecord
    """
    # writer
    sub_name = "%s_%d_%d_%s" % (raw_data_file_name, cnt, pidx, datetime.now().date())
    json_file_path, tfrecord_file_path = (
      os.path.join(workspace_dir, sub_name+x)
      for x in ['.json', '.tfrecord']
    )
    train_writer = TrainWriter(json_file_path, tfrecord_file_path)

    for conversation in data:
      conv_length = len(conversation)
      if (not self.is_pretrain) and (conv_length < 2):
        continue

      context_list, ner_texts_list = [], []
      for utterance_idx in range(conv_length):
        if not self.is_ner:
          a_text = conversation[utterance_idx]
          ner_texts = None
        else:
          a_text, a_words, a_keys = conversation[utterance_idx]
          ner_texts = self.zip_ner([a_words, a_keys])

        if self.is_pretrain or (utterance_idx > 0):
          outputs = self.transform_text_to_input_feature(
            a_text, context_list[-self.context_turn:], ner_texts_list[-self.context_turn:])
          if not outputs:
            break
          enc_in, dec_in, dec_out, masked_pos, utt_type, _ = outputs
          tf_example, json_example = self._create_example(enc_in, dec_in, dec_out, masked_pos, utt_type)
          train_writer.write(json_example, tf_example)

        context_list.append(a_text)
        ner_texts_list.append(ner_texts)

    train_writer.close()
    return

  def create_dataset(self, file_names=None):
    """
    1. load data
    2. aggregate multi-turn data
    3. assign <generate_wrapper> to Process
    4. join Process
    :param file_names: 특정 파일만 전처리하고 싶을 때 사용
    :return: None, save files
    """
    flag_objs = self.flag_objs
    num_process = flag_objs.num_process
    num_rand_mask = flag_objs.num_rand_mask if self.is_pretrain else 1

    # directory
    raw_data_file_dir, workspace_dir = self._get_directory()
    raw_data_files = file_names if file_names else os.listdir(raw_data_file_dir)

    for raw_data_file_name in raw_data_files:
      if self.verbose:
        print("[ %s ] is processing..." % raw_data_file_name)

      # load
      raw_data_path = os.path.join(raw_data_file_dir, raw_data_file_name)
      with open(raw_data_path, 'r') as fp:
        # ['A\n', 'B\n', 'A\n', '\n', 'A\n', ...]
        data = fp.readlines()

      # aggregate multi-turn data
      if self.is_pretrain:
        # [['A\n'], ['B\n'], ['A\n'], ['\n'], ['A\n'], ...]
        data = [[x] for x in data]
      elif self.is_ner:
        # [[['A\n', ['a', 'b'], [10, 132]], ['B\n', ...], ['A\n', ...]], [['A\n', ...], ...], ...]
        data = join_conversation(data=data, is_json=True)
      else:
        # [['A\n', 'B\n', 'A\n'], ['A\n', ...], ...]
        data = join_conversation(data=data)

      # data info
      len_data = len(data)
      sub_data_size = int(len_data/num_process)
      if self.verbose:
        print("  - total data length: %d" % len_data)
        print("  - sub data length: %d" % sub_data_size)

      #
      start_time = datetime.now()
      process_list = []
      for cnt in range(num_rand_mask):
        prev_idx = 0
        next_idx = sub_data_size
        copied_data = copy.deepcopy(data)

        for i in range(num_process+1):
          sub_data = copied_data[prev_idx:next_idx]
          prev_idx += sub_data_size
          next_idx += sub_data_size
          process_list.append(
            Process(
              target=self.generate_wrapper,
              args=(workspace_dir, raw_data_file_name, cnt, i, sub_data)
            )
          )

      if self.verbose:
        print('  - start process')
      for process in process_list:
        process.start()

      if self.verbose:
        print('  - join process')
      for process in process_list:
        process.join()

      if self.verbose:
        print('  - elapsed time: %s' % (datetime.now()-start_time))
    return

  def _preprocess_text(self, a_text, max_token_length, do_clip_front=True, do=False):
    """
    1. strip
    2. encode to idx
    3. check length
    :param a_text: 'text of utterance'
    :param max_token_length: max_seq_length - mandatory_token_length
    :param do_clip_front: 문장의 앞쪽 제거 (if True), 문장의 뒤쪽 제거 (if False)
    :param do: True면 None을 return하지 않고 끝까지 진행
    :return: encoded a_text or None
    """
    x = a_text.strip()
    if not do and len(x) < 1:
      return

    ids = self.tokenizer.encode(x)
    ids_len = len(ids)
    if not do and ids_len < self.flag_objs.min_seq_len:
      return

    if ids_len > max_token_length:
      ids = ids[-max_token_length:] if do_clip_front else ids[:max_token_length]

    return ids

  def _preprocess_ner(self, ner_text_list):
    """
    :param ner_text_list: [('안녕', 13), ('나', 132), ...] or None
    :return: [[1, 13], [2, 5, 132], ...] or None
    """
    if not ner_text_list:
      return

    ret = []
    for tup in ner_text_list:
      ids = self.tokenizer.encode(tup[0])
      ids.append(tup[1])
      if len(ids) > Constants.NER_UNIT_MAX:
        continue
      ret.append(ids)

    return ret

  def _to_pretrain_input_feature(self, ids):
    """
    id 형식의 utterance에 mask 취해 encoder/decoder의 input/output 형식으로 변경
    :param ids: id list | ex. [0, 1, 2, 5, ...]
    :return:
      enc_in: ids에 masking 추가 (=encoder_input)
      dec_in: dec_out의 이전 token (=decoder_input)
      dec_out: masking된 token의 실제 id (=decoder_output)
      masked_pos: masking 위치 (enc_in이랑 dec_out으로 원본 복구하려면 필요)
    """
    sub_seq_len = len(ids)
    block_size = self.flag_objs.block_size
    mask_prob = self.flag_objs.mask_prob

    # block 잡고 block 내에서 masking 위치 고르기
    positions = np.arange(0, sub_seq_len)
    masked_pos = []
    for i in range(1, sub_seq_len, block_size):
      block = positions[i: i + block_size]
      masked_len = max(1, int(len(block) * mask_prob))
      masked_block_start = np.random.choice(block[:len(block) - int(masked_len) + 1], 1)[0]
      masked_pos.extend(positions[masked_block_start: masked_block_start + masked_len])

    # masked_pos
    masked_pos = np.array(masked_pos)
    masked_pos_shift = np.array([p - 1 for p in masked_pos])

    # enc_in, dec_in, dec_out
    enc_in = np.array(ids)
    dec_in = np.take_along_axis(enc_in, masked_pos_shift, axis=0)
    dec_out = np.take_along_axis(enc_in, masked_pos, axis=0)
    replaced_tokens = self._replace_masked_tokens(dec_out)
    np.put(a=enc_in, ind=masked_pos, v=replaced_tokens)

    # to list
    enc_in = list(map(int, enc_in))
    dec_in = list(map(int, dec_in))
    dec_out = list(map(int, dec_out))
    masked_pos = list(map(int, masked_pos))

    # assertion
    assert len(enc_in) <= self.seq_len_encoder
    assert len(dec_in) <= self.seq_len_decoder
    assert len(dec_out) <= self.seq_len_decoder
    assert len(masked_pos) <= self.seq_len_decoder

    return enc_in, dec_in, dec_out, masked_pos

  def _to_finetune_input_feature(self, a_ids, context_ids_list, ner_ids_list=None):
    """
    list of ids 형식의 context와 ids 형식의 utterance를 input/output 형식으로 변경
    :param a_ids: id list | ex. [0, 1, 2, 5, ...]
    :param context_ids_list: id list | ex. [[0, 1, 2, 5, ...], ...]
    :param ner_ids_list: id list | ex. [[[1, 13], [2, 5, 132], ...], None, ...]
    :return:
      enc_in: context_ids_list 병합한 context 정보 + is_ner인 경우 앞쪽에 ner 정보 추가 (=encoder_input)
      dec_in: dec_out의 이전 token (=decoder_input)
      dec_out: 실제 answer인 a_ids에 special token 추가 (=decoder_output)
      utterance_type: multi-turn인 경우 대화 순서 indexing (최근 대화가 0)
    """
    # aggregate ids for encoder
    enc_in, utterance_type = self._squeeze_context(context_ids_list)

    # add bos/eos to ids for decoder
    dec_in = [self.bos_id] + a_ids
    dec_out = a_ids + [self.eos_id]

    # assertion
    assert len(enc_in) <= self.seq_len_encoder
    assert len(dec_in) <= self.seq_len_decoder
    assert len(dec_out) <= self.seq_len_decoder
    assert len(utterance_type) <= self.seq_len_encoder

    if not self.is_ner:
      return enc_in, dec_in, dec_out, utterance_type

    # aggregate ner
    ner_key = self._squeeze_ner(ner_ids_list)
    dec_in = [ner_key] + dec_in
    dec_out = [self.pad_id] + dec_out

    # assertion
    assert len(dec_in) <= self.seq_len_decoder
    assert len(dec_out) <= self.seq_len_decoder

    return enc_in, dec_in, dec_out, utterance_type

  @staticmethod
  def zip_ner(word_key_list):
    """
    :param word_key_list: [[word1, word2, ...], [key1, key2, ...]]
    :return: [(word1, key1), (word2, key2), ...] or None
    """
    if not word_key_list[0]:
      return
    return list(zip(*word_key_list))

  def _squeeze_context(self, ids_list):
    """
    :param ids_list: [[1, 2, ...], ...]
    :return: [..., 1, 2, ..., <sep>] (최근 발화를 앞쪽으로)
    """
    enc_in = ids_list[-1]
    utterance_type = []
    if self.flag_objs.is_multi_turn:
      enc_in += [self.sep_id]
      utterance_type = [0] * len(enc_in)
      for i, x in enumerate(reversed(ids_list[:-1])):
        enc_in += x + [self.sep_id]
        utterance_type += [i + 1] * (len(x) + 1)
      enc_in = enc_in[:self.seq_len_encoder]
      utterance_type = utterance_type[:self.seq_len_encoder]
    return enc_in, utterance_type

  @staticmethod
  def _squeeze_ner(ner_ids_list):
    """
    :param ner_ids_list: [[[1, 13], [2, 5, 132], ...], None, ...]
    :return: (int) top_ner_key
    """
    recent_ner = ner_ids_list[-1]
    if not recent_ner:
      return 0

    tier_dict = NER.get_info()
    ner_keys = [(x[-1], tier_dict[x[-1]]) for x in recent_ner if x[-1] != 0]
    if not ner_keys:
      return 0

    # top tier key 복수면 random 대신 max 선택
    min_tier = min([x[1] for x in ner_keys])
    top_ner_key = max([k for k, t in ner_keys if t == min_tier])
    return top_ner_key

  def _replace_masked_tokens(self, masked_tokens):
    """ for mask_random
    replace masked position to (mask or real or random)
      * p = (mask, real, random) = (80%, 10%, 10%)
    """
    len_mask = masked_tokens.shape[0]
    real_masked_tokens = masked_tokens
    rand_masked_tokens = random.sample(list(range(self.tokenizer.vocab_size)), k=len_mask)
    mask_masked_tokens = [self.tokenizer.get_mask_id()] * len_mask
    probs = np.random.choice(3, size=len_mask, p=[0.8, 0.1, 0.1])
    res = np.array(
      [np.where(probs == i, x, 0) for i, x in enumerate([mask_masked_tokens, real_masked_tokens, rand_masked_tokens])]
    ).sum(axis=0)

    return res

  def _create_example(self, enc_in, dec_in, dec_out, masked_pos=None, utt_type=None):
    """to json&tfrecord"""
    if self.is_pretrain and masked_pos is None:
      raise ValueError("When pretraining, position of masking should be provided.")
    if self.is_multi and utt_type is None:
      raise ValueError("When is_multi, utterance type should be provided.")

    json_example = {
      Constants.KEY.ENC_IN: enc_in,
      Constants.KEY.DEC_IN: dec_in,
      Constants.KEY.DEC_OUT: dec_out
    }

    tf_example = {
      Constants.KEY.ENC_IN: create_int_feature(enc_in),
      Constants.KEY.DEC_IN: create_int_feature(dec_in),
      Constants.KEY.DEC_OUT: create_int_feature(dec_out)
    }

    if self.is_pretrain:
      json_example[Constants.KEY.MASKED_POS] = masked_pos
      tf_example[Constants.KEY.MASKED_POS] = create_int_feature(masked_pos)

    if self.is_multi:
      json_example[Constants.KEY.UTT_TYPE] = utt_type
      tf_example[Constants.KEY.UTT_TYPE] = create_int_feature(utt_type)

    tf_example = tf.train.Example(features=tf.train.Features(feature=tf_example))

    return tf_example, json_example

  def _get_directory(self):
    #
    if self.is_pretrain:
      folder = self.flag_objs.pretrain_data_dir
    elif self.is_ner:
      folder = Constants.NER_DIR
    else:
      folder = Constants.CONVERSATION_DIR
    raw_data_file_dir = os.path.join(Constants.DATA_DIR, folder)

    #
    folder = Constants.PRETRAIN_DIR if self.is_pretrain else Constants.FINETUNE_DIR
    workspace_dir = os.path.join(Constants.DATA_DIR, folder, self.flag_objs.workspace)
    if not os.path.exists(workspace_dir):
      os.mkdir(workspace_dir)

    return raw_data_file_dir, workspace_dir


def create_int_feature(values):
  if type(values) is list:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  else:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def to_txt(dialog_list, path):
  """
  :param dialog_list: [['a', 'b', 'a'], ['c', 'd'], ...]
  :param path: path to save
  :return: None
  """
  f = open(path, 'w')
  for dialog in dialog_list:
    for sent in dialog:
      f.write('%s\n' % sent)
    f.write('\n')
  f.close()
  return


def join_conversation(path_to_file=None, data=None, is_json=False, do_strip=False):
  """
  :param path_to_file:
  :param data: ['A\n', 'B\n', 'A\n', '\n', 'A\n', ...]
  :param is_json:
  :param do_strip:
  :return:
    default: [['A\n', 'B\n', 'A\n'], ['A\n', ...], ...]
    do_strip: [['A', 'B', 'A'], ['A', ...], ...]
    is_json: [[['A\n', ['a', 'b'], [10, 132]], ['B\n', ...], ['A\n', ...]], [['A\n', ...], ...], ...]
  """
  if (path_to_file is None) and (data is None):
    raise ValueError("There is no data to join!")

  if data is None:
    with open(path_to_file, 'r') as f:
      data = f.readlines()

  if data[-1].strip() == '':
    data = data[:-1]

  ret = [[]]
  for line in data:
    line = line.strip()
    if line == '':
      ret.append([])
      continue

    # convert line by flag
    if do_strip:
      line = line.strip()
    elif is_json:
      line = json.loads(line)

    ret[-1].append(line)
  return ret


class TrainWriter(object):
  def __init__(self, json_file_path, tfrecord_file_path):
    super(TrainWriter, self).__init__()
    self.json_writer = tf.io.gfile.GFile(json_file_path, 'w')
    self.tf_writer = tf.io.TFRecordWriter(tfrecord_file_path)

  def write(self, json_example, tf_example):
    self.json_writer.write(json.dumps(json_example, ensure_ascii=False) + '\n')
    self.tf_writer.write(tf_example.SerializeToString())

  def close(self):
    self.json_writer.close()
    self.tf_writer.close()