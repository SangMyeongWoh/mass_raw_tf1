import os
import json
import numpy as np


PATH = '../data/ner'

total_length, total_cnt = [], 0
for file_name in os.listdir(PATH):
  # LOAD
  with open(os.path.join(PATH, file_name), 'r') as fp:
    dat = fp.readlines()
  parsed = [json.loads(x.strip()) for x in dat if x.strip()]

  # PRINT STATISTICS
  length = [len(x[1]) for x in parsed]
  cnt_zero = length.count(0)
  print('[ STATISTICS: %s ]' % file_name)
  print('  - mean of # NER per utterance: %.2f' % np.mean(length))
  print('  - min  of # NER per utterance: %d' % np.min(length))
  print('  - max  of # NER per utterance: %d' % np.max(length))
  print('  - std  of # NER per utterance: %.2f' % np.std(length))
  print('  - # of utterance without NER : %d (%.2f%%)' % (cnt_zero, cnt_zero / len(length) * 100))
  total_length.extend(length)
  total_cnt += cnt_zero

print('[ TOTAL ]')
print('  - mean of # NER per utterance: %.2f' % np.mean(total_length))
print('  - min  of # NER per utterance: %d' % np.min(total_length))
print('  - max  of # NER per utterance: %d' % np.max(total_length))
print('  - std  of # NER per utterance: %.2f' % np.std(total_length))
print('  - # of utterance without NER : %d (%.2f%%)' % (total_cnt, total_cnt / len(total_length) * 100))