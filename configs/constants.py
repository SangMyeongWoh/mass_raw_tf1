class Constants(object):
  class MEnum(object):
    @classmethod
    def has_value(cls, value):
      members = [getattr(cls, attr) for attr in dir(cls)
                 if not callable(getattr(cls, attr)) and not attr.startswith("__")]
      return value in members

    @classmethod
    def get_members(cls):
      members = [getattr(cls, attr) for attr in dir(cls)
                 if not callable(getattr(cls, attr)) and not attr.startswith("__")]
      return members

  # parameter - environment
  DATA_DIR = "./data"
  CHECKPOINT_DIR = "./checkpoints"
  TOKENIZER_DIR = "./tokenizer"

  # sub directory of DATA_DIR
  CLEAN_DIR = "clean"                          # cleaned data for training sentencepiece
  COMMENT_DATA = "comment"                     # comment data for training sentencepiece
  TOTAL_DATA = "total"                         # whole data for training sentencepiece
  NEWSDATA_FROMELASTIC0921 = "0921_tyr_newdata_fromelastic"
  WIKI_DIR = "wiki"                            # raw wiki data
  UTTERANCE_DIR = "utterance"                  # raw conversation data in english
  TRANSLATE_DIR = "translate"                  # preprocessed data of utterance and needs to be translated
  CONVERSATION_DIR = "conversation"            # conversation data for fine-tune
  NER_DIR = "ner"                              # conversation data with ner for fine-tune
  SENTENCEPIECE_DIR = "sentencepiece"          # preprocessed data to train sentencepiece
  SENTENCEPIECE_TYR_DIR = "sentencepiece_tyr"  # preprocessed data to train sentencepiece
  PRETRAIN_DIR = "pretrain"                    # preprocessed data to pre-train mass
  FINETUNE_DIR = "finetune"                    # preprocessed data to fine-tune mass

  # sub directory of CHECKPOINT_DIR
  PRETRAIN_CKP = "pretrain"
  FINETUNE_CKP = "finetune"

  # sub directory of UTTERANCE_DIR
  BLEND_DIR = "BlendedSkillTalk"            # Persona, Empathy, Depth combined
  CONV_DIR = "ConvAI2"                      # Persona
  MOVIE_DIR = "CornellMovieDialogsCorpus"   # Movie scripts
  EMPATHY_DIR = "EmpatheticDialogues"       # Empathy
  EMPATHY_DIR = "EmpatheticDialogues"       # Empathy
  UBUNTU_DIR = "UbuntuDialogueCorpus"       # Ubuntu chat
  WIZARD_DIR = "WizardOfWikipedia"          # Depth

  # parameter - tokenizer
  SENTENCEPIECE_DATA_NAME = "train_tyr.txt" # filename for train data
  SENTENCEPIECE_DATA_NAME_TEST = "xaa"      # filename for train data
  MAX_VOCAB_SIZE = 30000                    # it can enlarge network size
  TRAIN_DATA = "ko_news_fromelastic"                 # prefix for sentencepiece model
  SINGLE_TOK = "single_tok.txt"             # filename of single token

  # parameter - NER
  NER_HOST = "192.168.0.31"
  NER_PORT = 5001
  NER_KEY = "sentences"
  NER_UNIT_MAX = 10
  NER_PAD_ID = 0
  NER_SIZE = 150      # 0~132

  # parameter - model
  MAX_POSITION_EMBEDDINGS = 256   # TODO: 256으로 변경 (EX9,12 때문에 128로 설정)

  # parameter - fit
  NUM_CPU_THREADS = 8
  MAX_TO_KEEP = 5
  SAVE_CYCLE_STEP = 50000
  PRINT_CYCLE_STEP = 500

  # parameter - key of pretrain and finetune
  class KEY(MEnum):
    ORIGIN = "origin"
    ENC_IN = "encoder_input"
    DEC_IN = "decoder_input"
    DEC_OUT = "decoder_output"
    MASKED_POS = "masked_position"
    UTT_TYPE = "utterance_type"

  # parameter - type of utterance data
  class TYPE(MEnum):
    BLEND = 'blend'
    CONV = 'conv'
    MOVIE = 'movie'
    EMPATHY = 'empathy'
    UBUNTU = 'ubuntu'
    WIZARD = 'wizard'

  # parameter - model
  class MODEL(MEnum):
    VANILLA_TRANSFORMER = "vt"
    EVOLVED_TRANSFORMER = "et"
    NER_TRANSFORMER = "nt"

  # parameter - papago
  class PAPAGO(MEnum):
    # TODO: erase client id & secret before commit
    CLIENT_ID = "YOUR_CLIENT_ID"
    CLIENT_SECRET = "YOUR_CLIENT_SECRET"

  # parameter - api
  class API(MEnum):
    PREFIX_AMUMAL = ""
    GET = "sentence"
    KEY_Q = "question"
    KEY_A = "answer"
    KEY_C = "context"
    KEY_N = "ner"