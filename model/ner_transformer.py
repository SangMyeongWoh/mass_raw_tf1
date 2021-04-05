import tensorflow as tf
from configs.constants import Constants
from model.utils import get_shape_list
from model.transformer import (
  create_initializer,
  Embedding,
  PositionEmbedding,
  FinalLayer
)
from model.evolved_transformer import (
  EvolvedEncoder,
  EvolvedDecoder
)


"""
0. EmbeddingNERProcessor
"""


class EmbeddingWithNER(object):
  """Word Embedding Layer"""
  def __init__(self, vocab_size, ner_size, d_model):
    super(EmbeddingWithNER, self).__init__()
    self.vocab_size = vocab_size
    self.ner_size = ner_size
    self.d_model = d_model

  def __call__(self,
               input_ids,
               use_one_hot_embeddings=False,
               word_embedding_name="word_embeddings",
               ner_embedding_name="ner_embeddings"):

    embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[self.vocab_size, self.d_model],
      initializer=create_initializer())

    ner_table = tf.get_variable(
      name=ner_embedding_name,
      shape=[self.ner_size, self.d_model],
      initializer=create_initializer())

    concat_table = tf.concat((embedding_table, ner_table), axis=0)

    condition = tf.convert_to_tensor(
      [[True if i == 0 else False for i in range(input_ids.shape[1])]] * input_ids.shape[0])
    input_ids_plus = input_ids + self.vocab_size
    input_ids_plus = tf.where(condition, input_ids_plus, input_ids)

    if input_ids.shape.ndims == 2:
      input_ids = tf.expand_dims(input_ids, axis=[-1])
    input_shape = get_shape_list(input_ids)

    flat_input_ids = tf.reshape(input_ids_plus, [-1])

    if use_one_hot_embeddings:
      one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
      output = tf.matmul(one_hot_input_ids, concat_table)
    else:
      output = tf.gather(concat_table, flat_input_ids)

    output = tf.reshape(
      output, input_shape[0:-1] + [input_shape[-1] * self.d_model])
    return output, embedding_table


class EmbeddingNERProcessor(object):
  """NER embedding"""
  def __init__(self, vocab_size, ner_size, d_model, is_embed_pos, pe_encoder_len, pe_decoder_len):
    super(EmbeddingNERProcessor, self).__init__()
    self.d_model = d_model
    max_position_length = Constants.MAX_POSITION_EMBEDDINGS
    if (pe_encoder_len > max_position_length) or (pe_decoder_len > max_position_length):
      str1 = "Maximum length of encoder and decoder is '%d'." % max_position_length
      str2 = "The current encoder and decoder length are (%s, %s)." % (pe_encoder_len, pe_decoder_len)
      raise ValueError("%s %s" % (str1, str2))

    self.ne = EmbeddingWithNER(vocab_size, ner_size, d_model)
    self.we = Embedding(vocab_size, d_model)
    self.pe_encoder = PositionEmbedding(is_embed_pos, max_position_length, d_model)
    self.pe_decoder = PositionEmbedding(is_embed_pos, max_position_length, d_model)

  def __call__(self, x, is_embed_encoder):
    # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
    embedding_func = self.we if is_embed_encoder else self.ne
    x, self.embedding_table = embedding_func(x)

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.pe_encoder(x, "encoder_position_embeddings")

    return x


"""
1. NER transformer
"""


class NERTransformer(object):
  """named entity recognition transformer"""
  def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, vocab_size, type_size, ner_size,
               pe_encoder_len, pe_decoder_len, rate=0.1, is_embed_pos=False, is_finetune=False):
    super(NERTransformer, self).__init__()
    self.d_model = d_model
    self.is_finetune = is_finetune    # if finetune, use pe_encoder only

    self.embedding = EmbeddingNERProcessor(
      vocab_size, ner_size, d_model, is_embed_pos, pe_encoder_len, pe_decoder_len)

    self.utterance_type_embedding = Embedding(type_size, d_model)

    self.encoder = EvolvedEncoder(num_encoder_layers, d_model, num_heads, dff, rate)

    self.decoder = EvolvedDecoder(num_decoder_layers, d_model, num_heads, dff, rate)

    self.final_layer = FinalLayer(d_model, vocab_size)

  def __call__(self, inp, tar, training, enc_padding_mask, look_ahead_mask,
               dec_padding_mask, conv_mask, scope='evolved', use_utt_type=False, utt_type=None):
    with tf.variable_scope(scope):
      with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        # (batch_size, inp_seq_len, d_model)
        emb_enc = self.embedding(inp, is_embed_encoder=True)
        emb_dec = self.embedding(tar, is_embed_encoder=False)
        if use_utt_type:
          emb_type, _ = self.utterance_type_embedding(utt_type, word_embedding_name="utterance_type_embeddings")
          emb_enc += emb_type

      with tf.variable_scope('encoder'):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(emb_enc, training, enc_padding_mask)

      with tf.variable_scope('decoder'):
        # (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
          emb_dec, enc_output, training, look_ahead_mask, dec_padding_mask, conv_mask)

      with tf.variable_scope('output'):
        # (batch_size, tar_seq_len, vocab_size)
        final_output = self.final_layer(dec_output, self.embedding.embedding_table)

    return final_output, attention_weights
