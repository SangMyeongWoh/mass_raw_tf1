import os
import json
import urllib.request
from tqdm import tqdm

from configs.constants import Constants
from src.data_handler import join_conversation, to_txt


class TranslateTask(object):
  """Translate English to Korean using Papago API"""
  def __init__(self, flag_objs, verbose=True):
    self.flag_objs = flag_objs
    self.verbose = verbose

  def translate_api(self, text):
    request = urllib.request.Request("https://naveropenapi.apigw.ntruss.com/nmt/v1/translation")
    request.add_header("X-NCP-APIGW-API-KEY-ID", Constants.PAPAGO.CLIENT_ID)
    request.add_header("X-NCP-APIGW-API-KEY", Constants.PAPAGO.CLIENT_SECRET)

    enc_text = urllib.parse.quote(text)    # 뭔지 모르겠지만 되네..
    data = "source=en&target=ko&text=" + enc_text
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    res_code = response.getcode()
    if res_code == 200:
      ret_json = json.loads(response.read().decode('utf-8'))
      ret = ret_json['message']['result']['translatedText']
      return ret
    else:
      if self.verbose:
        print('*Error in Translator.api_wrapper. Error Code:', res_code)
      return False

  def run(self):
    route = os.path.join(Constants.DATA_DIR, Constants.TRANSLATE_DIR)

    for file_name in os.listdir(route):
      if self.verbose:
        print('#' * 50)
        print('<%s> is processing...' % file_name)

      write_name = file_name.split('_')[0]
      write_path = os.path.join(Constants.DATA_DIR, Constants.CONVERSATION_DIR, write_name)
      if os.path.exists(write_path):
        print('<%s> is already translated. SKIP translation!' % file_name)
        continue

      path_to_file = os.path.join(route, file_name)
      dialogs = join_conversation(path_to_file=path_to_file)

      translated, cnt = [], 0
      for dialog in tqdm(dialogs):
        dialog_t = []
        for utterance in dialog:
          utterance_t = self.translate_api(utterance)
          if not utterance_t:
            # 번역이 안되면 그 뒤 발화는 제거
            cnt += 1
            break
          else:
            dialog_t.append(utterance_t)

        if len(dialog_t) > 1:
          translated.append(dialog_t)

      to_txt(translated, write_path)

    return