import os
import sys
import pdb

sys.path.append('..')

from utils.sentencepiece_wrapper import SentencePieceWrapper
from configs.constants import Constants

tokenizer = SentencePieceWrapper(
  model_dir=os.path.join('../', Constants.TOKENIZER_DIR),
  model_name="./2020-06-18_wiki_30000.model"
)

print('--\nvocab_size: %d\n--' % tokenizer.vocab_size)
pdb.set_trace()