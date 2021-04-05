import sentencepiece as spm
import os
from datetime import date


class SentencePieceWrapper(object):
  """SentencePieceWrapper for Korean"""
  def __init__(self,
               model_dir="./tokenizer",
               model_name="2020-04-27_wiki_10000.model",
               is_load=True,
               is_append_single_tok=False,
               single_token="single_tok.txt"):
    """if is_load: load fitted sentencepiece model"""
    self.sp = spm.SentencePieceProcessor()
    self.is_append_single_tok = is_append_single_tok
    self.single_token_route = os.path.join(model_dir, single_token)

    self._pad = "[PAD]"
    self._unk = "[UNK]"
    self._bos = "[BOS]"
    self._eos = "[EOS]"
    self._mask = "[MASK]"
    self._cls = "[CLS]"
    self._sep = "[SEP]"

    if is_load:
      # load sentence piece model
      path = os.path.join(model_dir, model_name)
      self.sp.Load(path)
      self.vocab_size = self.sp.get_piece_size()

  def encode(self, x):
    """
    encode to ids
    :param x: string to encode
    :return: [id, id, id, id, ...]
    """
    return self.sp.encode_as_ids(x)

  def decode(self, x):
    """
    decode from ids
    :param x: list of ids
    :return: "decoded text"
    """
    return self.sp.decode_ids(x)

  def decode_by_token(self, x):
    """
    decode token by token from ids
    :param x: list of ids
    :return: "token | token | ..."
    """
    ret = [self.sp.decode_ids([_id]) for _id in x]
    return " | ".join(ret)

  def fit(self, input_name, wroute, input_category, vocab_size):
    """
    [PAD], [MASK], [CLS], [SEP] 추가하여 sentencepiece 학습
    :param input_name: 학습데이터 경로/이름.txt
    :param wroute: 학습된 모델 저장 위치
    :param input_category: 학습데이터의 종류 (wiki, conversation, ...)
    :param vocab_size:
    :return:
    """
    prefix = './%s/%s_%s_%d' % (wroute, date.today(), input_category, vocab_size)

    command = '--input=%s --vocab_size=%d --model_prefix=%s ' % (input_name, vocab_size, prefix)
    command += '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    command += '--pad_piece=%s --unk_piece=%s --bos_piece=%s --eos_piece=%s ' % (
      self._pad, self._unk, self._bos, self._eos)
    command += '--user_defined_symbols=%s,%s,%s' % (self._mask, self._cls, self._sep)
    command += '--mining_sentence_size=100000'
    if self.is_append_single_tok:
      # is_append_single_tok이 True면 vocab에 single_tok 제외한 다른 애들은 안 나옴
      # 추측으로는 token 분리할 때 user_defined_symbols가 우선순위가 있기 때문인듯...?
      added = self.handle_single_tok()
      command += ',%s' % added
      command += ' --hard_vocab_limit=false'
    command += ' --max_sentence_length=4096 --character_coverage=1.0 '
    command += '--input_sentence_size=5000000'
    command += '--shuffle_input_sentence=true'

    spm.SentencePieceTrainer.train(command)

    self.sp = spm.SentencePieceProcessor()
    self.sp.Load('%s.model' % prefix)
    return

  def handle_single_tok(self):
    """extract single token for sentencepiece dictionary"""
    with open(self.single_token_route, 'r') as f:
      dat = f.readlines()

    # dat에 중복 있어서 set 취해줌
    added = ','.join([x.strip() for x in set(dat)])
    return added

  def get_pad_id(self):
    """pad id = 0"""
    return self.sp.pad_id()

  def get_unk_id(self):
    """unk id = 1"""
    return self.sp.unk_id()

  def get_bos_id(self):
    """bos id = 2"""
    return self.sp.bos_id()

  def get_eos_id(self):
    """eos id = 3"""
    return self.sp.eos_id()

  def get_mask_id(self):
    """default mask id = 4"""
    return self.sp.piece_to_id(self._mask)

  def get_sep_id(self):
    return self.sp.piece_to_id(self._sep)

  def get_cls_id(self):
    return self.sp.piece_to_id(self._cls)