import json
import os

path_dir ='/home/lab/mass_tyr/data/sunurimal/'

file_list = os.listdir(path_dir)

with open(path_dir + file_list[0]) as json_file:
  json_data = json.load(json_file)

speaker_id = -999
text = None

# for utterance in json_data['document'][0]["utterance"]:
#   if speaker_id != utterance['speaker_id']:
#     print(text)
#     speaker_id = utterance['speaker_id']
#     text = utterance['original_form']
#   else:
#     text = text + utterance['original_form']


for i in range(len(file_list)):
  with open(path_dir + file_list[i]) as json_file:
    json_data = json.load(json_file)
    speaker_id = -999
    text = ''
    for utterance in json_data['document'][0]["utterance"]:
      if speaker_id != utterance['speaker_id']:
        print(text)
        speaker_id = utterance['speaker_id']
        text = utterance['original_form']

      else:
        text = text + utterance['original_form']

    print(text)

  print(i)
  print("\n\n\n")