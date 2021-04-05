import tensorflow as tf
from model.transformer import (
  create_initializer,
  Dense,
  LayerNormalization,
  Dropout,
  MultiHeadAttention,
  PointWiseFeedForwardNetwork,
  EmbeddingPostProcessor,
  FinalLayer,
  Embedding
)


"""
0. base - GLU, Conv1D, SepConv1D, SwishFeedForwardNetwork
"""


class GLU(object):
  """GLU"""
  def __init__(self, d_model):
    super(GLU, self).__init__()

    self.dense1 = Dense(d_model)
    self.dense2 = Dense(d_model, activation=tf.nn.sigmoid)

  def __call__(self, x):
    values = self.dense1(x)
    gates = self.dense2(x)
    x = values * gates
    return x


class Conv1D(object):
  """Conv1D"""
  def __init__(self, d_model_in, d_model_out, kernel_size=3):
    super(Conv1D, self).__init__()
    self.filter_shape = [kernel_size, d_model_in, d_model_out]

  def __call__(self, x, name="conv_filter"):
    conv_filter = tf.get_variable(name=name, shape=self.filter_shape,
                                  initializer=create_initializer())
    x = tf.nn.conv1d(x, conv_filter, stride=1, padding="SAME")
    return x


class SepConv1D(object):
  """use tf.nn.separable_conv2d"""
  def __init__(self, d_model_in, d_model_out, kernel_size=9):
    super(SepConv1D, self).__init__()
    channel_multiplier = 1      # keras default setting
    self.depth_shape = [kernel_size, 1, d_model_in, channel_multiplier]
    self.point_shape = [1, 1, d_model_in*channel_multiplier, d_model_out]

  def __call__(self, x, padding="SAME"):
    depthwise_filter = tf.get_variable(name="depthwise_filter", shape=self.depth_shape,
                                       initializer=create_initializer())
    pointwise_filter = tf.get_variable(name="pointwise_filter", shape=self.point_shape,
                                       initializer=create_initializer())

    x = tf.expand_dims(x, 2)      # [B, S, D] --> [B, S, 1, D]
    x = tf.nn.separable_conv2d(x, depthwise_filter, pointwise_filter,
                               strides=[1, 1, 1, 1], padding=padding)
    x = tf.squeeze(x, 2)          # [B, S, 1, D] --> [B, S, D]
    return x


class SwishFeedForwardNetwork(object):
  """feed forward with swish activation"""
  def __init__(self, d_model, dff):
    super(SwishFeedForwardNetwork, self).__init__()
    self.first = Dense(dff, activation=tf.nn.swish)
    self.second = Dense(d_model)

  def __call__(self, x):
    x = self.first(x)     # (batch_size, seq_len, dff)
    x = self.second(x)    # (batch_size, seq_len, d_model)
    return x


"""
1. encoder and decoder block
"""


class EncoderBlock(object):
  """encoder block of the evolved transformer"""
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderBlock, self).__init__()
    self.d_model = d_model
    self.dff = dff
    self.dhh = int(d_model/2)

    self.pad_len = dff - self.dhh

    self.glu = GLU(d_model)
    self.dense = Dense(dff, activation=tf.nn.relu)
    self.conv = Conv1D(d_model, self.dhh, 3)
    self.sep_conv = SepConv1D(dff, self.dhh, 9)
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.layernorm3 = LayerNormalization(epsilon=1e-6)
    self.layernorm4 = LayerNormalization(epsilon=1e-6)
    self.layernorm5 = LayerNormalization(epsilon=1e-6)
    self.layernorm6 = LayerNormalization(epsilon=1e-6)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    self.dropout3 = Dropout(rate)
    self.dropout4 = Dropout(rate)
    self.dropout5 = Dropout(rate)
    self.dropout6 = Dropout(rate)

  def __call__(self, x, training, mask):
    # about convolution mask: (batch_size, 1, 1 seq_len) --> (batch_size, seq_len)
    nonpadding = 1.0 - tf.squeeze(mask, [1, 2])
    mask1 = tf.tile(tf.expand_dims(nonpadding, 2), [1, 1, self.d_model])
    mask2 = tf.tile(tf.expand_dims(nonpadding, 2), [1, 1, self.dff])

    with tf.variable_scope("gated_linear_unit"):
      # (batch_size, seq_len, d_model)
      x = self.layernorm1(x)
      out1 = self.glu(x)
      out1 = self.dropout1(out1, training=training)
      out1 = self.layernorm2(x + out1)
      out1 *= mask1

    with tf.variable_scope("conv_left"):
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, dff)
      left = self.dense(out1)
      left = self.dropout2(left, training=training)

    with tf.variable_scope("conv_right"):
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, dhh)
      right = self.conv(out1)
      right = tf.nn.relu(right)
      right = self.dropout3(right, training=training)

    with tf.variable_scope("conv_hidden"):
      # (batch_size, seq_len, dff)
      hidden = left + tf.pad(right, [[0, 0], [0, 0], [0, self.pad_len]])
      hidden = self.layernorm3(hidden)
      hidden *= mask2

      # (batch_size, seq_len, dhh) --> (batch_size, seq_len, d_model)
      out2 = self.sep_conv(hidden)
      out2 = tf.pad(out2, [[0, 0], [0, 0], [0, self.dhh]])
      out2 = self.dropout4(out2, training=training)
      out2 = self.layernorm4(out1 + out2)

    with tf.variable_scope("attention"):
      # (batch_size, seq_len, d_model)
      out3, _ = self.mha(out2, out2, out2, mask)
      out3 = self.dropout5(out3, training=training)
      out3 = self.layernorm5(out2 + out3)

    with tf.variable_scope("feed_forward"):
      # (batch_size, seq_len, dff) --> (batch_size, seq_len, d_model)
      out4 = self.ffn(out3)
      out4 = self.dropout6(out4, training=training)
      out4 = self.layernorm6(out3 + out4)

    return out4


class DecoderBlock(object):
  """decoder block of the evolved transformer"""
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderBlock, self).__init__()
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

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    self.dropout3 = Dropout(rate)
    self.dropout4 = Dropout(rate)
    self.dropout5 = Dropout(rate)
    self.dropout6 = Dropout(rate)
    self.dropout7 = Dropout(rate)
    self.dropout8 = Dropout(rate)

  def __call__(self, x, enc_output, training,
               look_ahead_mask, padding_mask, conv_mask):
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

    with tf.variable_scope("encoder_decoder"):
      # (batch_size, seq_len, d_model)
      out4, attn_weights_block4 = self.mha4(
        enc_output, enc_output, out3, padding_mask)
      out4 = self.dropout7(out4, training=training)
      out4 = self.layernorm6(out3 + out4)

    with tf.variable_scope("feed_forward"):
      # (batch_size, seq_len, d_model)
      out5 = self.ffn(out4)
      out5 = self.dropout8(out5, training=training)
      out5 = self.layernorm7(out4 + out5)

    return out5, attn_weights_block1, attn_weights_block2, attn_weights_block3, attn_weights_block4


"""
2. evolved encoder and decoder
"""


class EvolvedEncoder(object):
  """encoder of the evolved transformer"""
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(EvolvedEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.enc_layers = [EncoderBlock(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = Dropout(rate)

  def __call__(self, x, training, mask):
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      with tf.variable_scope("layer_%d" % i):
        x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


class EvolvedDecoder(object):
  """decoder of the evolved transformer"""
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(EvolvedDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [DecoderBlock(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = Dropout(rate)

  def __call__(self, x, enc_output, training,
               look_ahead_mask, padding_mask, conv_mask):
    attention_weights = {}

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      with tf.variable_scope("layer_%d" % i):
        x, block1, block2, block3, block4 = self.dec_layers[i](
          x, enc_output, training, look_ahead_mask, padding_mask, conv_mask)

      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
      attention_weights['decoder_layer{}_block3'.format(i + 1)] = block3
      attention_weights['decoder_layer{}_block4'.format(i + 1)] = block4

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


"""
3. evolved transformer
"""


class EvolvedTransformer(object):
  """evolved transformer"""
  def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, vocab_size, type_size,
               pe_encoder_len, pe_decoder_len, rate=0.1, is_embed_pos=False, is_finetune=False):
    super(EvolvedTransformer, self).__init__()
    self.d_model = d_model
    self.is_finetune = is_finetune    # if finetune, use pe_encoder only

    self.embedding = EmbeddingPostProcessor(
      vocab_size, d_model, is_embed_pos, pe_encoder_len, pe_decoder_len)

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
        emb_dec = self.embedding(tar, is_embed_encoder=self.is_finetune)
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