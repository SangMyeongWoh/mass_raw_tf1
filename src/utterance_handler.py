import os
import json
import numpy as np
from datetime import datetime
from ast import literal_eval

from configs.constants import Constants
from src.data_handler import to_txt


class PreprocessUtteranceTask(object):
  def __init__(self, flag_objs, verbose=True):
    self.flag_objs = flag_objs
    self.verbose = verbose
    self.route = os.path.join(Constants.DATA_DIR, Constants.UTTERANCE_DIR)

  def run(self, do_all=True):
    """run single task (at most six tasks)"""
    # effective when do_all is false
    do_blend = True
    do_conv = True
    do_movie = False
    do_empathy = False
    do_wizard = True

    if do_all:
      self.single_task(Constants.TYPE.BLEND)
      self.single_task(Constants.TYPE.CONV)
      self.single_task(Constants.TYPE.MOVIE)
      self.single_task(Constants.TYPE.EMPATHY)
      self.single_task(Constants.TYPE.WIZARD)
      return

    if do_blend:
      self.single_task(Constants.TYPE.BLEND)
    if do_conv:
      self.single_task(Constants.TYPE.CONV)
    if do_movie:
      self.single_task(Constants.TYPE.MOVIE)
    if do_empathy:
      self.single_task(Constants.TYPE.EMPATHY)
    if do_wizard:
      self.single_task(Constants.TYPE.WIZARD)
    return

  def single_task(self, task_type):
    """
    flow of single task
      1. open file / merge file / extract necessaries
      2. separate speaker / aggregate dialog
      3. check integrity (more than one utterance for each speaker)
      4. save preprocessed file

    features
      - blend: clean dataset
      - conv: emoticon included dataset
      - movie: weird encoding
      - empathy: clean dataset
      - ubuntu: too informal dataset (skip)
      -
    """
    full_name, merge_function, write_name = {
      Constants.TYPE.BLEND: [Constants.BLEND_DIR, self.blend_task, 'blend_eng'],
      Constants.TYPE.CONV: [Constants.CONV_DIR, self.conv_task, 'conv_eng'],
      Constants.TYPE.MOVIE: [Constants.MOVIE_DIR, self.movie_task, 'movie_eng'],
      Constants.TYPE.EMPATHY: [Constants.EMPATHY_DIR, self.empathy_task, 'empathy_eng'],
      Constants.TYPE.UBUNTU: [Constants.UBUNTU_DIR, self.ubuntu_task, 'ubuntu_eng'],
      Constants.TYPE.WIZARD: [Constants.WIZARD_DIR, self.wizard_task, 'wizard_eng']
    }[task_type]

    tic = datetime.now()
    if self.verbose:
      print('[%s]' % full_name)

    # merge and extract
    merge_tic = datetime.now()
    merged = merge_function()
    if self.verbose:
      print('  - merge and extract: %s' % (datetime.now() - merge_tic))

    # check
    checked, cnt, cnt_empty = [], 0, 0
    check_tic = datetime.now()
    for dialog in merged:
      preprocessed, prev_id = [], None
      for id_, sent in dialog:
        if len(sent) < 1:       # sent가 ''인 경우 건너뛰기
          cnt_empty += 1
          continue

        elif id_ != prev_id:    # 화자가 달라지면 새로운 발화로 저장
          preprocessed.append(sent)

        else:                   # 화자가 같으면 이전 발화에 이어 붙이기
          preprocessed[-1] += ' ' + sent

        # 발화 추가했으면 prev_id 변경
        prev_id = id_

      # 전처리 종료 후 발화 2개 이상이면 저장
      if len(preprocessed) < 2:
        cnt += 1
        continue
      checked.append(preprocessed)

    if self.verbose:
      print('  - check the order of utterance: %s' % (datetime.now() - check_tic))
      len_ = len(checked)
      pr_ = cnt / len_ * 100
      print('    * cnt_not_pair: %d (%.1f%% = %d/%d)' % (cnt, pr_, cnt, len_))
      print('    * cnt_emtpy_utterance: %d' % cnt_empty)

    # save
    save_tic = datetime.now()
    path = os.path.join(Constants.DATA_DIR, Constants.TRANSLATE_DIR, write_name)
    to_txt(dialog_list=checked, path=path)
    if self.verbose:
      print('  - save the pre-processed file: %s' % (datetime.now() - save_tic))
      print('  - total processing time: %s' % (datetime.now() - tic))
    return

  def blend_task(self):
    """preprocess BlendedSkillTalk to merged

    raw data: list of dict (json type)
      dict has keys>
        personas, context_dataset, free_turker_utterance, guided_turker_utterance,
        additional_context, dialog, suggestions, suggestion_orders, chosen_suggestions,
        workers, bad_workers, hit_ids, assignment_ids

      example of dialog>
        [
          [
            [0, 'sentence'],
            [1, 'sentence'],
            [0, 'sentence'],
            ...
          ],
          ...
        ]
    """
    merged = []
    for file_name in ['train.json', 'valid.json', 'test.json']:
      path = os.path.join(self.route, Constants.BLEND_DIR, file_name)
      with open(path, 'r') as f:
        dat = json.loads(f.readline())
        extracted = [x.get('dialog') for x in dat]
        merged.extend(extracted)
    return merged

  def conv_task(self):
    """preprocess ConvAI2
      - data_tolokers.json: (3127 dialogues) July 2-8 2018
      - data_intermediate.json: (291 dialogues) July 9 to October 29 2018
      - data_volunteers.json: (1111 dialogues) October 29 to December 17 2018

    raw data: list of dict (json type)
      dict has keys>
        dialog, start_time, end_time, bot_profile, user_profile, eval_score,
        profile_match, participant1_id, participant2_id

      example of dialog>
        [
          {_id: ~, sender: ~, text: ~, evaluation_score: ~, sender_class: ~J},
          ...
        ]
    """
    merged = []
    for file_name in ['data_intermediate.json', 'data_tolokers.json', 'data_volunteers.json']:
      path = os.path.join(self.route, Constants.CONV_DIR, file_name)
      with open(path, 'r') as f:
        dat = json.loads(f.readline())
        extracted = [[[z.get('sender'), z.get('text')] for z in y] for y in (x.get('dialog') for x in dat)]
        merged.extend(extracted)
    return merged

  def movie_task(self):
    """preprocess CornellMovieDialogsCorpus
      - movie_conversations.txt: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
      - movie_lines.txt: L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    """
    with open(os.path.join(self.route, Constants.MOVIE_DIR, 'movie_conversations.txt'), 'r') as f:
      conversations = f.readlines()
    with open(os.path.join(self.route, Constants.MOVIE_DIR, 'movie_lines.txt'), 'rb') as f:
      # 'rb' due to encoding error: \xad (soft-hyphens)
      lines = [x.decode('ascii', 'ignore') for x in f.readlines()]

    sep = ' +++$+++ '
    # use literal_eval instead of json due to encoding issue
    conversations = [literal_eval(x.strip().split(sep)[-1]) for x in conversations]
    # use strip after split due to the case when text == '\n'
    lines = {lineID: [c_id, text.strip()] for lineID, c_id, movieID, c_name, text in (x.split(sep) for x in lines)}
    merged = [[lines[y] for y in x] for x in conversations]
    return merged

  def empathy_task(self):
    """preprocess EmpatheticDialogues
      - train.csv, valid.csv, test.csv
      - header is in the first row of each file
        :: conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags
      - convert '_comma_' to ','
      - numpy is faster than pandas
    """
    merged = []
    for file_name in ['train.csv', 'valid.csv', 'test.csv']:
      path = os.path.join(self.route, Constants.EMPATHY_DIR, file_name)
      with open(path, 'r') as f:
        dat = [x.replace('\n', '').split(',')[:8] for x in f.readlines()]
        arr = np.array(dat[1:])
        id_ = list(set(arr[:, 0]))
        for k in id_:
          sample = arr[arr[:, 0] == k]
          sample = sample[sample[:, 1].argsort()]
          utterance = [x.replace('_comma_', ',').strip() for x in sample[:, 5]]
          dialog = list(zip(sample[:, 4], utterance))
          merged.append(dialog)

    return merged

  @staticmethod
  def ubuntu_task():
    """번역하면 너무 이상해져서 skip"""
    return []

  def wizard_task(self):
    """preprocess BlendedSkillTalk to merged

    raw data: list of dict (json type)
      dict has keys>
        chosen_topic, persona, wizard_eval, dialog, chosen_topic_passage

      dialog has keys>
        speaker, text, checked_sentence, checked_passage, retrieved_passages, retrieved_topics
    """
    # 추측으로는 data.json이 전체 데이터인 것 같아 data.json만 사용
    merged = []
    for file_name in ['data.json']:
      path = os.path.join(self.route, Constants.WIZARD_DIR, file_name)
      with open(path, 'r') as f:
        dat = json.loads(f.readline())
        extracted = [[[z['speaker'], z['text']] for z in y] for y in (x.get('dialog') for x in dat)]
        merged.extend(extracted)
    return merged
