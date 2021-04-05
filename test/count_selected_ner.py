import os
import json
from tqdm import tqdm


PATH = '../data/finetune/EX3_single_ner'

cnt = {i: 0 for i in range(133)}
for file_name in tqdm(os.listdir(PATH)):
  if file_name.endswith('.tfrecord'):
    continue

  # LOAD
  file_to_path = os.path.join(PATH, file_name)
  with open(file_to_path, 'r') as f:
    dat = f.readlines()

  for x in dat:
    stripped = x.strip()
    if not stripped:
      continue
    parsed = json.loads(stripped)
    cnt[parsed['decoder_input'][0]] += 1

print(json.dumps(cnt, indent=2, sort_keys=True))