import tensorflow as tf
import numpy as np
from configs.constants import Constants
from model.utils import get_shape_list


"""
0. base
"""


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


class Dense(object):
  """Dense"""
  def __init__(self, d_model, activation=None):
    super(Dense, self).__init__()
    self.d_model = d_model
    self.activation = activation

  def __call__(self, x):
    x = tf.cast(x, tf.float32)
    x = tf.compat.v1.layers.dense(
      x,
      self.d_model,
      activation=self.activation,
      kernel_initializer=create_initializer()
    )
    return x


class LayerNormalization(object):
  """LayerNormalization"""
  def __init__(self, epsilon=1e-6):
    super(LayerNormalization, self).__init__()
    self.epsilon = epsilon

  def __call__(self, x, name=None):
    return tf.contrib.layers.layer_norm(
      inputs=x, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


class Dropout(object):
  """Dropout"""
  def __init__(self, rate):
    super(Dropout, self).__init__()
    self.rate = rate

  def __call__(self, x, training):
    if self.rate is None or self.rate == 0.0:
      return x

    output = tf.nn.dropout(x, 1.0 - self.rate)
    return output


class Embedding(object):
  """Word Embedding Layer"""
  def __init__(self, vocab_size, d_model):
    super(Embedding, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model

  def __call__(self,
               input_ids,
               use_one_hot_embeddings=False,
               word_embedding_name="word_embeddings"):

    embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[self.vocab_size, self.d_model],
      initializer=create_initializer())

    if input_ids.shape.ndims == 2:
      input_ids = tf.expand_dims(input_ids, axis=[-1])

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
      one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
      output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
      output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(
      output, input_shape[0:-1] + [input_shape[-1] * self.d_model])
    return output, embedding_table


"""
1. positional encoding & position embedding
"""


def get_angles(pos, i, d_model):
  """from tensorflow homepage"""
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  """fixed positional encoding"""
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def position_embedding(
  input_tensor,
  max_position_embeddings=Constants.MAX_POSITION_EMBEDDINGS,
  position_embedding_name="position_embeddings"):
  """trainable position embedding"""
  input_shape = get_shape_list(input_tensor, expected_rank=3)   # (batch_size, seq_len, d_model)
  seq_length = input_shape[1]
  width = input_shape[2]

  assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
  with tf.control_dependencies([assert_op]):

    full_position_embeddings = tf.get_variable(
      name=position_embedding_name,
      shape=[max_position_embeddings, width],
      initializer=create_initializer())

    position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])

    num_dims = len(input_tensor.shape.as_list())
    position_broadcast_shape = []
    for _ in range(num_dims - 2):
      position_broadcast_shape.append(1)

    position_broadcast_shape.extend([seq_length, width])

    position_embeddings = tf.reshape(
      position_embeddings, position_broadcast_shape)  # 1 * max_seq * embedding_size
    input_tensor += position_embeddings

  return input_tensor


class PositionEmbedding(object):
  """if is_embed_pos: position embedding
     else position encoding"""
  def __init__(self, is_embed_pos, max_position_embeddings, d_model):
    super(PositionEmbedding, self).__init__()
    self.flag = is_embed_pos
    self.max_len = max_position_embeddings
    self.pos_weights = positional_encoding(self.max_len, d_model)

  def __call__(self, x, name="position_embeddings"):
    if self.flag:
      x = position_embedding(x, self.max_len, position_embedding_name=name)
    else:
      seq_len = tf.shape(x)[1]
      x += self.pos_weights[:, :seq_len, :]

    return x


"""
2. masking
"""


def create_padding_mask(seq):
  """PAD의 index는 0"""
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  """decoder에서 사용"""
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask, dec_target_padding_mask


"""
3. scaled dot product attention
"""


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


"""
4. multi-head attention
"""


class MultiHeadAttention(object):
  """MultiHeadAttention"""
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)
    self.dense = Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def __call__(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
      q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


"""
5. point wise feed forward network
"""


class PointWiseFeedForwardNetwork(object):
  """기본 feed forward"""
  def __init__(self, d_model, dff):
    super(PointWiseFeedForwardNetwork, self).__init__()
    self.first = Dense(dff, activation=tf.nn.relu)
    self.second = Dense(d_model)

  def __call__(self, x):
    x = self.first(x)     # (batch_size, seq_len, dff)
    x = self.second(x)    # (batch_size, seq_len, d_model)
    return x


"""
6. encoder and decoder
"""


class EncoderLayer(object):
  """vanilla encoder layer"""
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)

  def __call__(self, x, training, mask):
    with tf.variable_scope("attention"):
      attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
      attn_output = self.dropout1(attn_output, training=training)
      out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    with tf.variable_scope("feed_forward"):
      ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
      ffn_output = self.dropout2(ffn_output, training=training)
      out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class DecoderLayer(object):
  """vanilla decoder layer"""
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.layernorm3 = LayerNormalization(epsilon=1e-6)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    self.dropout3 = Dropout(rate)

  def __call__(self, x, enc_output, training,
               look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    with tf.variable_scope("attention"):
      attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
      attn1 = self.dropout1(attn1, training=training)
      out1 = self.layernorm1(attn1 + x)

    with tf.variable_scope("encoder_decoder"):
      attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
      attn2 = self.dropout2(attn2, training=training)
      out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    with tf.variable_scope("feed_forward"):
      ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
      ffn_output = self.dropout3(ffn_output, training=training)
      out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2


class Encoder(object):
  """vanilla encoder"""
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = Dropout(rate)

  def __call__(self, x, training, mask):
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      with tf.variable_scope("layer_%d" % i):
        x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


class Decoder(object):
  """vanilla decoder"""
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = Dropout(rate)

  def __call__(self, x, enc_output, training,
               look_ahead_mask, padding_mask):
    attention_weights = {}

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      with tf.variable_scope("layer_%d" % i):
        x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                               look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


"""
7. transformer
"""


class Transformer(object):
  """vanilla transformer"""
  def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
               pe_encoder_len, pe_decoder_len, rate=0.1, is_embed_pos=False, is_finetune=False):
    super(Transformer, self).__init__()
    self.d_model = d_model
    self.is_finetune = is_finetune    # if finetune, use pe_encoder only

    self.embedding = EmbeddingPostProcessor(
      vocab_size, d_model, is_embed_pos, pe_encoder_len, pe_decoder_len)

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate)

    self.final_layer = FinalLayer(d_model, vocab_size)

  def __call__(self, inp, tar, training, enc_padding_mask,
               look_ahead_mask, dec_padding_mask, scope='transformer'):
    with tf.variable_scope(scope):
      with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        # (batch_size, inp_seq_len, d_model)
        emb_enc = self.embedding(inp, is_embed_encoder=True)
        emb_dec = self.embedding(tar, is_embed_encoder=self.is_finetune)

      with tf.variable_scope('encoder'):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(emb_enc, training, enc_padding_mask)

      with tf.variable_scope('decoder'):
        # (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
          emb_dec, enc_output, training, look_ahead_mask, dec_padding_mask)

      with tf.variable_scope('output'):
        # (batch_size, tar_seq_len, vocab_size)
        final_output = self.final_layer(dec_output, self.embedding.embedding_table)

    return final_output, attention_weights


class EmbeddingPostProcessor(object):
  """Wrapped Embedding Layer for Transformer"""
  def __init__(self, vocab_size, d_model, is_embed_pos, pe_encoder_len, pe_decoder_len):
    super(EmbeddingPostProcessor, self).__init__()
    self.d_model = d_model
    max_position_length = Constants.MAX_POSITION_EMBEDDINGS
    if (pe_encoder_len > max_position_length) or (pe_decoder_len > max_position_length):
      str1 = "Maximum length of encoder and decoder is '%d'." % max_position_length
      str2 = "The current encoder and decoder length are (%s, %s)." % (pe_encoder_len, pe_decoder_len)
      raise ValueError("%s %s" % (str1, str2))

    self.we = Embedding(vocab_size, d_model)
    self.pe_encoder = PositionEmbedding(is_embed_pos, max_position_length, d_model)
    self.pe_decoder = PositionEmbedding(is_embed_pos, max_position_length, d_model)

  def __call__(self, x, is_embed_encoder=True, do_pos_embed=True):
    x, self.embedding_table = self.we(x)   # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    if not do_pos_embed:
      return x

    if is_embed_encoder:
      x = self.pe_encoder(x, "encoder_position_embeddings")
    else:
      x = self.pe_decoder(x, "decoder_position_embeddings")
    return x


class FinalLayer(object):
  """Output Layer for Transformer"""
  def __init__(self, d_model, vocab_size):
    super(FinalLayer, self).__init__()
    self.final_layer = Dense(d_model)
    self.vocab_size = vocab_size

  def __call__(self, x, embedding_table):
    """
    :param x: input tensor
    :param embedding_table: embedding table from Embedding
    :return:
    """
    embed_bias = tf.get_variable(
        "embed_bias",
        shape=[self.vocab_size],
        initializer=tf.zeros_initializer())

    x = self.final_layer(x)  # (batch_size, seq_len, d_model)
    x = tf.matmul(x, embedding_table, transpose_b=True)  # (batch_size, seq_len, vocab_size)
    x = tf.nn.bias_add(x, embed_bias)
    return x
