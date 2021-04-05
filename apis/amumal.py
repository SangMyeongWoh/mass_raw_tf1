from flask import Blueprint, request, jsonify
from absl import flags
import requests
import time

from configs.constants import Constants
from utils.api_tool import wrap_output
from src.amumal_model import AMUMAL
from apis.manager import APIManager


flag_objs = flags.FLAGS
task = AMUMAL(flag_objs=flag_objs, is_training=False, mode='f')
manager = APIManager()
amumal_blueprint = Blueprint("amumal", __name__,)


@amumal_blueprint.before_request
def before_request():
  pass


@amumal_blueprint.after_request
def after_request(response):
  return response


@amumal_blueprint.route("/", methods=["GET"])
def answer_to_utterance():
  """단일 발화에 대해 답변 생성"""
  utterance = request.args.get(Constants.API.GET)

  out = task.generate(q_text=utterance,
                      do_sample=flag_objs.do_sample,
                      top_k=flag_objs.top_k)

  result_dict = {
    Constants.API.KEY_Q: utterance,
    Constants.API.KEY_A: out
  }

  if flag_objs.is_ner:
    ner_url = "http://%s:%d/?sentence=%s" % (Constants.NER_HOST, Constants.NER_PORT, utterance)
    ner_res = requests.get(ner_url).json()['results'][0]['entities']

    result_dict[Constants.API.KEY_N] = ner_res

  return wrap_output(result_dict)

@amumal_blueprint.route("/post", methods=["POST"])
def answer_to_utterances():
  """단일 발화에 대해 답변 생성"""
  utterances = request.get_json()

  result = []

  star_time = time.time()
  for utterance in utterances['sentence']:
    out = task.generate(q_text=utterance,
                        do_sample=flag_objs.do_sample,
                        top_k=flag_objs.top_k)
    result_dict = {
      Constants.API.KEY_Q: utterance,
      Constants.API.KEY_A: out
    }
    result.append(result_dict)

  #return utterances
  result.append({"time": time.time() - star_time})
  return wrap_output(result)


@amumal_blueprint.route("/key/", methods=["GET"])
def issue_chat_key():
  """chat id 발급"""
  chat_id_dict = manager.generate_key()
  return wrap_output(chat_id_dict)


@amumal_blueprint.route("/<int:chat_id>/", methods=["GET"])
def answer_to_context(chat_id):
  """chat_id와 현재까지 나눴던 대화내용을 기반으로 답변 생성"""
  utterance = request.args.get(Constants.API.GET)

  manager.update_conversation(chat_id, utterance)
  context = manager.get_conversation(chat_id, max_context=flag_objs.max_multi_turn)

  out = task.generate(q_text=context,
                      do_sample=flag_objs.do_sample,
                      top_k=flag_objs.top_k)

  manager.update_conversation(chat_id, out)
  result_dict = {
    Constants.API.KEY_C: context,
    Constants.API.KEY_A: out
  }
  return wrap_output(result_dict)