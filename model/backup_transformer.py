import tensorflow as tf
from configs.constants import Constants
from model.transformer import (
  Embedding,
  MultiHeadAttention,
  LayerNormalization,
  Dropout,
  Encoder,
  EmbeddingPostProcessor,
  FinalLayer
)
from model.evolved_transformer import (
  SepConv1D,
  SwishFeedForwardNetwork,
  EvolvedEncoder
)


"""
0. NER padding mask
"""


def create_ner_padding_mask(seq):
  """PAD의 index는 0"""
  seq = tf.cast(tf.math.equal(seq, Constants.NER_PAD_ID), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


"""
1. decoder block with NER
"""


class DecoderBlockNER(object):
  """decoder block of the evolved transformer"""
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderBlockNER, self).__init__()
    self.d_model = d_model
    self.dff = dff
    self.dhh = int(d_model/2)

    self.sep_conv1_size = 11
    self.sep_conv2_size = 7
    self.sep_conv3_size = 7

    self.mha1 = MultiHeadAttention(d_model, num_heads*2)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.mha3 = MultiHeadAttention(d_model, num_heads)
    self.mha4 = MultiHeadAttention(d_model, num_heads)
    self.mha5 = MultiHeadAttention(d_model, num_heads)

    self.sep_conv1 = SepConv1D(d_model, d_model*2, self.sep_conv1_size)
    self.sep_conv2 = SepConv1D(d_model, self.dhh, self.sep_conv2_size)
    self.sep_conv3 = SepConv1D(d_model*2, d_model, self.sep_conv3_size)

    self.ffn = SwishFeedForwardNetwork(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.layernorm3 = LayerNormalization(epsilon=1e-6)
    self.layernorm4 = LayerNormalization(epsilon=1e-6)
    self.layernorm5 = LayerNormalization(epsilon=1e-6)
    self.layernorm6 = LayerNormalization(epsilon=1e-6)
    self.layernorm7 = LayerNormalization(epsilon=1e-6)
    self.layernorm8 = LayerNormalization(epsilon=1e-6)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    self.dropout3 = Dropout(rate)
    self.dropout4 = Dropout(rate)
    self.dropout5 = Dropout(rate)
    self.dropout6 = Dropout(rate)
    self.dropout7 = Dropout(rate)
    self.dropout8 = Dropout(rate)
    self.dropout9 = Dropout(rate)

  def __call__(self, x, enc_output, ner_output, training,
               look_ahead_mask, padding_mask, conv_mask, ner_mask):
    # about convolution mask: (batch_size, 1, 1 seq_len) --> (batch_size, seq_len)
    nonpadding = 1.0 - tf.squeeze(conv_mask, [1, 2])
    mask1 = tf.tile(tf.expand_dims(nonpadding, 2), [1, 1, self.d_model])
    mask2 = tf.tile(tf.expand_dims(nonpadding, 2), [1, 1, self.d_model*2])

    with tf.variable_scope("preprocess"):
      # (batch_size, seq_len, d_model)
      x = self.layernorm1(x)

    with tf.variable_scope("attention_left"):
      # (batch_size, seq_len, d_model)
      attn_left, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
      attn_left = self.dropout1(attn_left, training=training)

    with tf.variable_scope("attention_right"):
      # (batch_size, seq_len, d_model)
      attn_right, attn_weights_block2 = self.mha2(enc_output, enc_output, x, padding_mask)
      attn_right = self.dropout2(attn_right, training=training)

    with tf.variable_scope("attention_hidden"):
      # (batch_size, seq_len, d_model)
      out1 = self.layernorm2(x + attn_left + attn_right)
      out1 *= mask1

    with tf.variable_scope("conv_left"):
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model*2)
      left_state = tf.pad(out1, [[0, 0], [self.sep_conv1_size - 1, 0], [0, 0]])
      conv_left = self.sep_conv1(left_state, padding="VALID")
      conv_left = tf.nn.relu(conv_left)
      conv_left = self.dropout3(conv_left, training=training)

    with tf.variable_scope("conv_right"):
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, dhh)
      right_state = tf.pad(out1, [[0, 0], [self.sep_conv2_size - 1, 0], [0, 0]])
      conv_right = self.sep_conv2(right_state, padding="VALID")
      conv_right = self.dropout4(conv_right, training=training)

    with tf.variable_scope("conv_hidden"):
      # (batch_size, seq_len, d_model*2)
      conv_right = tf.pad(conv_right, [[0, 0], [0, 0], [0, self.d_model*2 - self.dhh]])
      hidden = conv_left + conv_right
      hidden = self.layernorm3(hidden)
      hidden *= mask2

      # (batch_size, seq_len, d_model*2) --> (batch_size, seq_len, d_model)
      hidden = tf.pad(hidden, [[0, 0], [self.sep_conv3_size - 1, 0], [0, 0]])
      out2 = self.sep_conv3(hidden, padding="VALID")
      out2 = self.dropout5(out2, training=training)
      out2 = self.layernorm4(out1 + out2)

    with tf.variable_scope("attention"):
      # (batch_size, seq_len, d_model)
      out3, attn_weights_block3 = self.mha3(out2, out2, out2, look_ahead_mask)
      out3 = self.dropout6(out3, training=training)
      out3 = self.layernorm5(out2 + out3)

    with tf.variable_scope("ner_encoder_decoder"):
      # (batch_size, seq_len, d_model)
      out4, attn_weights_block4 = self.mha4(
        ner_output, ner_output, out3, ner_mask)
      out4 = self.dropout7(out4, training=training)
      out4 = self.layernorm6(out3 + out4)

    with tf.variable_scope("encoder_decoder"):
      # (batch_size, seq_len, d_model)
      out5, attn_weights_block5 = self.mha5(
        enc_output, enc_output, out4, padding_mask)
      out5 = self.dropout8(out5, training=training)
      out5 = self.layernorm7(out4 + out5)

    with tf.variable_scope("feed_forward"):
      # (batch_size, seq_len, d_model)
      out6 = self.ffn(out5)
      out6 = self.dropout9(out6, training=training)
      out6 = self.layernorm8(out5 + out6)

    return out6


"""
2. evolved decoder with NER
"""


class EvolvedDecoderNER(object):
  """decoder of the evolved transformer"""
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(EvolvedDecoderNER, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [DecoderBlockNER(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = Dropout(rate)

  def __call__(self, x, enc_output, ner_output, training,
               look_ahead_mask, padding_mask, conv_mask, ner_mask):
    attention_weights = {}    # dummy

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      with tf.variable_scope("layer_%d" % i):
        x = self.dec_layers[i](x, enc_output, ner_output, training,
                               look_ahead_mask, padding_mask, conv_mask, ner_mask)

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


"""
3. NER transformer
"""


class NERTransformer(object):
  """named entity recognition transformer"""
  def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, vocab_size, type_size, ner_size,
               pe_encoder_len, pe_decoder_len, rate=0.1, is_embed_pos=False, is_finetune=False):
    super(NERTransformer, self).__init__()
    self.d_model = d_model
    self.is_finetune = is_finetune    # if finetune, use pe_encoder only

    self.embedding = EmbeddingPostProcessor(
      vocab_size, d_model, is_embed_pos, pe_encoder_len, pe_decoder_len)
    self.utterance_type_embedding = Embedding(type_size, d_model)
    self.ner_embedding = Embedding(ner_size, d_model)

    self.encoder = EvolvedEncoder(num_encoder_layers, d_model, num_heads, dff, rate)

    self.ner_encoder = Encoder(1, d_model, num_heads, dff, rate)
    self.layernorm = LayerNormalization(epsilon=1e-6)

    self.decoder = EvolvedDecoderNER(num_decoder_layers, d_model, num_heads, dff, rate)

    self.final_layer = FinalLayer(d_model, vocab_size)

  def __call__(self, inp, tar, ner_w, ner_k, training, enc_padding_mask, look_ahead_mask,
               dec_padding_mask, conv_mask, scope='evolved', use_utt_type=False, utt_type=None):
    with tf.variable_scope(scope):
      with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        # (batch_size, inp_seq_len, d_model)
        emb_enc = self.embedding(inp, is_embed_encoder=True)
        emb_dec = self.embedding(tar, is_embed_encoder=self.is_finetune)
        if use_utt_type:
          emb_type, _ = self.utterance_type_embedding(utt_type, word_embedding_name="utterance_type_embeddings")
          emb_enc += emb_type

        # (batch_size, ner_seq_len, d_model)
        emb_ner_w = self.embedding(ner_w, do_pos_embed=False)
        emb_ner_k, _ = self.ner_embedding(ner_k, word_embedding_name="ner_embeddings")

      with tf.variable_scope('encoder'):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(emb_enc, training, enc_padding_mask)

      with tf.variable_scope('ner_encoder'):
        # (batch_size, ner_seq_len, d_model)
        ner_mask = create_ner_padding_mask(ner_w)
        emb_ner_inp = self.layernorm(emb_ner_w + emb_ner_k)
        ner_output = self.ner_encoder(emb_ner_inp, training, ner_mask)

      with tf.variable_scope('decoder'):
        # (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
          emb_dec, enc_output, ner_output, training, look_ahead_mask, dec_padding_mask, conv_mask, ner_mask)

      with tf.variable_scope('output'):
        # (batch_size, tar_seq_len, vocab_size)
        final_output = self.final_layer(dec_output, self.embedding.embedding_table)

    return final_output, attention_weights