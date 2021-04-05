import os
import sys
import json

sys.path.append('..')

from utils.sentencepiece_wrapper import SentencePieceWrapper
from configs.constants import Constants

tokenizer = SentencePieceWrapper(
  model_dir=os.path.join('../', Constants.TOKENIZER_DIR),
  model_name="./2020-06-18_wiki_30000.model"
)


def tester(mode, workspace):
  if mode == 'p':
    sub_dir = Constants.PRETRAIN_DIR
    # dat_list = ['clean_clien', 'clean_wiki', 'blend']
    dat_list = ['raw_nia', 'empathy', 'blend']
  elif mode == 'f':
    sub_dir = Constants.FINETUNE_DIR
    dat_list = ['data', 'raw_nia', 'raw_flagship', 'blend', 'conv', 'empathy', 'wizard']
  else:
    raise ValueError("Unexpected mode: %s ('p' and 'f' are allowed)" % mode)

  print('===== MODE: %s =====' % mode)
  cnt = {x: 0 for x in dat_list}
  data_length = []
  route = os.path.join('../', Constants.DATA_DIR, sub_dir, workspace)
  file_names = os.listdir(route)
  file_length = len(file_names)
  for i, file_name in enumerate(file_names):
    if file_name.endswith('.tfrecord'):
      continue

    flag_list = [file_name.startswith(x) for x in dat_list]
    if True not in flag_list:
      continue

    with open(os.path.join(route, file_name), 'r') as f:
      dat = [json.loads(x) for x in f.readlines()]

    _ind = flag_list.index(True)
    cnt[dat_list[_ind]] += len(dat)
    data_length.extend([len(x.get(Constants.KEY.ENC_IN)) for x in dat])

    _ith = 5
    print('-- (%d / %d)' % (i, file_length))
    print('[ ENCODER INPUT ] %s' % tokenizer.decode(dat[_ith].get(Constants.KEY.ENC_IN)))
    print('[ DECODER INPUT ] %s' % tokenizer.decode(dat[_ith].get(Constants.KEY.DEC_IN)))
    print('[ DECODER OUTPUT ] %s' % tokenizer.decode(dat[_ith].get(Constants.KEY.DEC_OUT)))

  print('--')
  print('[ MAX LENGTH OF ENCODER INPUT ] %d' % max(data_length))
  print('[ MIN LENGTH OF ENCODER INPUT ] %d' % min(data_length))
  print('[ TOTAL DATA SIZE ] {:,d}'.format(sum(cnt.values())))
  print(json.dumps(cnt, indent=2, sort_keys=True))
  return


if __name__ == '__main__':
  tester(mode='f', workspace='EX2_multi_turn')