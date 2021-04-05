import os
import json
import requests
from tqdm import tqdm

from configs.constants import Constants


class NERHandler(object):
  """
  conversation data에 대해 NER 돌리는 작업
  """

  def __init__(self, flag_objs):
    """NER"""
    super(NERHandler, self).__init__()
    self.flag_objs = flag_objs
    self.verbose = True

  def apply_ner(self, file_names=None):
    """
    conversation data 불러와서 ner api를 이용해 적용 후 저장
    :return: json 형식
      [
        [
          "스펀지밥의 캐릭터를 표현하기 위해 힐렌버그는 로코의 모던 라이프에서 그와 함께 일했던 톰 케니에게 다가갔다."
          ["로코", "모던 라이프", "톰 케니"],
          [132, 132, 1]
        ],
        ...,
        [],   # conversation separator
        ...
      ]
    """
    from src.data_handler import join_conversation

    # PATH
    route = os.path.join(Constants.DATA_DIR, Constants.CONVERSATION_DIR)
    file_names = file_names if file_names else os.listdir(route)
    wroute = os.path.join(Constants.DATA_DIR, Constants.NER_DIR)
    if not os.path.exists(wroute):
      os.mkdir(wroute)

    # REQUESTS
    ner_url = "http://%s:%d" % (Constants.NER_HOST, Constants.NER_PORT)
    ner_key = Constants.NER_KEY

    # PROCESS
    for file_name in file_names:
      if self.verbose:
        print("[ %s ] is processing..." % file_name)

      # LOAD
      with open(os.path.join(route, file_name), 'r') as fp:
        dat = fp.readlines()
      dat = join_conversation(data=dat, do_strip=True)

      json_writer = open(os.path.join(wroute, file_name), 'w')

      for i in tqdm(range(len(dat))):
        res = requests.post(ner_url, data=json.dumps({ner_key: dat[i]})).json()
        for x in res['results']:
          a_json = parse_ner(x)
          json_writer.write(json.dumps(a_json, ensure_ascii=False) + '\n')
        json_writer.write('\n')
      json_writer.close()

    return


def get_ner(txt):
  ner_url = "http://%s:%d/?sentence=%s" % (Constants.NER_HOST, Constants.NER_PORT, txt)
  res = requests.get(ner_url).json()['results'][0]
  parsed = parse_ner(res)
  return parsed


def parse_ner(ner_result):
  sentence_ = ner_result['sentence']
  ent = ner_result['entities']
  word_ = [x['word'] for x in ent] if ent else []
  key_ = [x['key'] for x in ent] if ent else []
  return [sentence_, word_, key_]