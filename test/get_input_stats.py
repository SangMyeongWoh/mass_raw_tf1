import os
import json
import numpy as np

PATH = '../data/finetune/%s' % ['EX1_single_turn', 'EX2_multi_turn', 'EX3_single_ner'][1]
MODE, LEN = [
  ('encoder_input', 256),
  ('decoder_input', 64)
][0]
print('*** PATH: %s ***' % PATH)
print('*** Confirm <%s> with length "%d" ***' % (MODE, LEN))

data_list = ['raw_flagship', 'raw_nia', 'data', 'conv', 'empathy', 'wizard', 'blend']
aggregated = {x: [] for x in data_list}

for file_name in os.listdir(PATH):
  # IGNORE tfrecord FILE (count json file)
  if file_name.endswith('.tfrecord'):
    continue

  #
  k = data_list[[file_name.startswith(x) for x in data_list].index(True)]

  # LOAD
  file_to_path = os.path.join(PATH, file_name)
  with open(file_to_path, 'r') as f:
    dat = f.readlines()

  for x in dat:
    x_json = json.loads(x)
    nonzero_len = np.count_nonzero(x_json[MODE])
    aggregated[k].append(nonzero_len)

stats = []
for k, v in aggregated.items():
  print('[ %s ]' % k)
  length1 = v.count(LEN)
  length2 = sum(v)
  length3 = len(v)
  print('  # of data with length %d: %d (%.2f%%)' % (LEN, length1, length1 / length3 * 100))
  print('  average length: %.1f' % (length2 / length3))
  stats.append([length1, length2, length3])

print('[ total ]')
length1, length2, length3 = [sum(x) for x in list(zip(*stats))]
print('  # of data with length %d: %d (%.2f%%)' % (LEN, length1, length1 / length3 * 100))
print('  average length: %.1f' % (length2 / length3))

# import pdb; pdb.set_trace()