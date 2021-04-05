import os
import re

from utils.sentencepiece_wrapper import SentencePieceWrapper
from configs.constants import Constants
from langdetect import detect_langs, detect


class BuildTokenizerTask(object):
  """Tokenizer인 sentencespiece를 학습하는 클래스"""
  def __init__(self, flag_objs):
    """SentencePieceWrapper를 is_load False로 불러 새롭게 학습할 준비"""
    self.tokenizer = SentencePieceWrapper(
      model_dir=Constants.TOKENIZER_DIR,
      is_load=False
    )
    self.is_load_single_tok = True
    self.flag_objs = flag_objs

  def fit(self):
    """학습 데이터가 없으면 생성 후 sentencepiece 학습"""
    input_name = os.path.join(
      Constants.DATA_DIR,
      Constants.SENTENCEPIECE_TYR_DIR,
      Constants.SENTENCEPIECE_DATA_NAME
    )

    if not os.path.exists(input_name):
      print("there is no data")
      self.preprocess_data()



    self.tokenizer.fit(
      input_name=input_name,
      wroute=Constants.TOKENIZER_DIR,
      input_category=Constants.TRAIN_DATA,
      vocab_size=Constants.MAX_VOCAB_SIZE
    )
    return



  def preprocess_data(self):
    """clien과 wiki 데이터를 sentencepiece 학습할 수 있게 전처리하는 함수"""
    route = os.path.join(Constants.DATA_DIR, Constants.NEWSDATA_FROMELASTIC0921)
    wroute = os.path.join(Constants.DATA_DIR, Constants.SENTENCEPIECE_TYR_DIR)
    if not os.path.exists(wroute):
      os.mkdir(wroute)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    fw = open(os.path.join(wroute, Constants.SENTENCEPIECE_DATA_NAME), 'w')
    for _file in os.listdir(route):
      with open(os.path.join(route, _file), 'r') as fr:
        dat = fr.readlines()

      for line in dat:

        ### 언어 체크 ###
        try:
          if detect(line) != "ko":
            continue
        except:
          print("language is not available")
          continue

        ### 텍스트 길이 체크 ###
        if len(line) < 2 or len(line) > self.flag_objs.seq_len_encoder:
          continue

        ### url 체크 ###
        if "http" in line:
          continue

        ### 한자 포함된 글자 제거 ###
        if re.search(u'[\u4e00-\u9fff]', line):
          continue

        ### 이모티콘 제거 ###
        line = emoji_pattern.sub(r'', line)

        if len(line.strip()) > 0:
          fw.write(line)

    if self.is_load_single_tok:
      # 학습데이터에 single token 추가
      added = self.tokenizer.handle_single_tok()
      added = list(set(added.split(',')))
      for x in added:
        if len(x.strip()) > 0:
          fw.write("%s\n" % x)

    fw.close()
    return

