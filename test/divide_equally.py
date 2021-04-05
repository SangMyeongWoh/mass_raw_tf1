import os
import json
from datetime import datetime
import tensorflow as tf


def get_total_length():
  aggregated = []
  for file_name in os.listdir(PATH):
    # IGNORE tfrecord FILE (count json file)
    if file_name.endswith('.tfrecord'):
      continue

    # LOAD
    file_to_path = os.path.join(PATH, file_name)
    with open(file_to_path, 'r') as f:
      dat = f.readlines()

    aggregated.append([file_name, len(dat)])
  return sum([x[1] for x in aggregated])


def create_int_feature(values):
  if type(values) is list:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  else:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def get_tf_writer(idx):
  sub_name = "wiki_clien_%.3d_%s.tfrecord" % (idx, datetime.now().date())
  tf_writer = tf.io.TFRecordWriter(os.path.join(ROUTE, sub_name))
  return sub_name, tf_writer


def divide_tf():
  cnt, idx = 0, 0
  sub_name, tf_writer = get_tf_writer(idx)

  for file_name in os.listdir(PATH):
    # wiki & clien json files only
    if file_name.endswith('.tfrecord') or not file_name.startswith('clean'):
      continue

    # load
    print('[ %s ] is processing...' % file_name)
    file_to_path = os.path.join(PATH, file_name)
    with open(file_to_path, 'r') as f:
      dat = f.readlines()

    # txt to json
    for x in dat:
      x_json = json.loads(x)
      x_tf = {
        "origin": create_int_feature(x_json['origin']),
        "encoder_input": create_int_feature(x_json['encoder_input']),
        "decoder_input": create_int_feature(x_json['decoder_input']),
        "decoder_output": create_int_feature(x_json['decoder_output']),
        "masked_position": create_int_feature(x_json['masked_position'])
      }
      x_tf = tf.train.Example(features=tf.train.Features(feature=x_tf))

      tf_writer.write(x_tf.SerializeToString())
      cnt += 1

      if cnt == DATA_PER_FILE:
        print('  >> writing [ %s ] is finished with length %d!' % (sub_name, cnt))
        tf_writer.close()
        cnt = 0
        idx += 1
        if idx >= FILE_LENGTH:
          break
        sub_name, tf_writer = get_tf_writer(idx)
    if idx >= FILE_LENGTH:
      break

  return


if __name__ == '__main__':
  PATH = '../data/pretrain/EX1'

  TOTAL_LENGTH = get_total_length()   # 9669595
  FILE_LENGTH = 200
  DATA_PER_FILE = int(TOTAL_LENGTH / FILE_LENGTH)

  ROUTE = '../data/pretrain/EX3'
  if not os.path.exists(ROUTE):
    os.mkdir(ROUTE)

  divide_tf()