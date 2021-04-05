import os
import sys

sys.path.append('..')

from utils.sentencepiece_wrapper import SentencePieceWrapper
from configs.constants import Constants

tokenizer = SentencePieceWrapper(
  model_dir=os.path.join('../', Constants.TOKENIZER_DIR),
  model_name="./2020-09-08_ko_comment_30000.model"
)

print('--\nvocab_size: %d\n--' % tokenizer.vocab_size)


sent_list = [
  "안녕하세요 <s> </s>",
  "안녕하세요 [PAD] [MASK]",
  "' ` ´ ‘ ’ “ ” ",
  "! \" # $ % & ' ( ) * + , - . / ",
  "0123456789",
  ": ; < = > ? @",
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "[\]^_`",
  "a b c d e f g h i j k l m n o p q r s t u v w x y z",
  "{ | } ~",
  "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ",
  "ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ",
  "ㄲ ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅄ",
  "가갉",
  "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄲㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ훰휵휸흇흖흽홴횅횹?",
  "ㄱ".encode('utf-8')
]

for sent in sent_list:
  out_ids = tokenizer.encode(sent)
  out_sent = tokenizer.decode(out_ids)
  print("[origin] ", sent)
  print("[ids   ] ",out_ids)
  print("[sent  ] ",out_sent)
  print("--")

sent = tokenizer.decode([21568, 6664, 833, 93, 843, 17893, 58])
print(sent)
print(tokenizer.get_pad_id())


print('=' * 20 + '\n')
with open('../tokenizer/single_tok.txt') as fp:
  single_tok = fp.readlines()

for st in single_tok:
  out_ids = tokenizer.encode(st)
  if tokenizer.get_unk_id() in out_ids:
    out_sent = tokenizer.decode(out_ids)
    print("[origin] ", st)
    print("[ids   ] ", out_ids)
    print("[sent  ] ", out_sent)
    print("--")