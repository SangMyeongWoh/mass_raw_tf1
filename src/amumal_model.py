import os
import json
import tensorflow as tf
import numpy as np

from configs.constants import Constants
import configs.model_params as model_params
from src.data_handler import HandleDatasetTask
from model.transformer import create_masks, Transformer
from model.evolved_transformer import EvolvedTransformer
from model.ner_transformer import NERTransformer
from model.utils import (
  make_session, get_features_from_tfrecords, get_shape_list,
  loss_function, accuracy_function, handle_init_checkpoint,
  train_optimizer, get_assignment_map_from_checkpoint
)


class AMUMAL(object):
  """main task of AMUMAL: pretrain / finetune / generate"""
  def __init__(self, flag_objs, is_training, mode):
    """load model params"""
    super(AMUMAL, self).__init__()
    self.flag_objs = flag_objs
    self.params = model_params.PARAMS_MAP[flag_objs.param_set]
    self.is_training = is_training
    self.is_ner = flag_objs.is_ner
    self.is_multi = flag_objs.is_multi_turn
    self.verbose = True
    self.is_saver_to_load = False

    # for vocab size and evaluation
    self.data_handler = HandleDatasetTask(flag_objs, mode)
    self.tokenizer = self.data_handler.tokenizer
    self.vocab_size = self.tokenizer.vocab_size

    # about model
    self.sess = make_session(flag_objs.target_gpu, flag_objs.gpu_usage)
    if mode in ['p', 'pretrain', 'f', 'finetune']:
      self._build_model(mode)
    else:
      raise ValueError('Wrong input for mode: %s (only "p" and "f" are allowed)' % mode)

    # initialize variables
    if not is_training:
      self._restore_checkpoint()

  def _build_model(self, mode):
    """build AMUMAL model"""
    is_pretrain = mode in ['p', 'pretrain']
    model_type = self.flag_objs.model_type

    # Define encoder / decoder sequence length
    enc_len = self.flag_objs.seq_len_encoder
    dec_len = [
      self.flag_objs.seq_len_decoder,
      int(enc_len * self.flag_objs.mask_prob)
    ][is_pretrain]
    if self.verbose:
      print(' [ MODE ] %s' % mode)
      print(' [ MODEL TYPE ] %s' % model_type)
      print(' [ LENGTH OF ENCODER ] %d' % enc_len)
      print(' [ LENGTH OF DECODER ] %d' % dec_len)
      if self.is_multi:
        print(' [ MAX MULTI TURN ] %d' % self.flag_objs.max_multi_turn)

    # Define parser
    def _parser(tfrecord):
      """tfrecord parser"""
      parsing_feature = {
        Constants.KEY.ENC_IN: tf.io.FixedLenFeature([enc_len], tf.int64),
        Constants.KEY.DEC_IN: tf.io.FixedLenFeature([dec_len], tf.int64),
        Constants.KEY.DEC_OUT: tf.io.FixedLenFeature([dec_len], tf.int64)
      }

      if is_pretrain:
        parsing_feature[Constants.KEY.MASKED_POS] = tf.io.FixedLenFeature([dec_len], tf.int64)
      if self.is_multi:
        parsing_feature[Constants.KEY.UTT_TYPE] = tf.io.FixedLenFeature([enc_len], tf.int64)

      example = tf.io.parse_single_example(tfrecord, parsing_feature)
      return example

    # Make input features from tfrecord
    sub_dir = Constants.PRETRAIN_DIR if is_pretrain else Constants.FINETUNE_DIR
    data_dir = os.path.join(Constants.DATA_DIR, sub_dir, self.flag_objs.preprocess_workspace)
    tfrecord_file_path = [
      os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith('.tfrecord')]
    features, _, input_initializer = get_features_from_tfrecords(
      tfrecord_file_path, _parser, self.is_training, self.flag_objs)
    self.input_initializer = input_initializer

    # Get features that needs to make AMUMAL model
    self.encoder_input = encoder_input = tf.placeholder_with_default(
      features[Constants.KEY.ENC_IN],
      shape=get_shape_list(features[Constants.KEY.ENC_IN]),
      name=Constants.KEY.ENC_IN
    )

    self.decoder_input = decoder_input = tf.placeholder_with_default(
      features[Constants.KEY.DEC_IN],
      shape=get_shape_list(features[Constants.KEY.DEC_IN]),
      name=Constants.KEY.DEC_IN
    )

    self.decoder_output = decoder_output = tf.placeholder_with_default(
      features[Constants.KEY.DEC_OUT],
      shape=get_shape_list(features[Constants.KEY.DEC_OUT]),
      name=Constants.KEY.DEC_OUT
    )

    if is_pretrain:
      self.masked_position = tf.placeholder_with_default(
        features[Constants.KEY.MASKED_POS],
        shape=get_shape_list(features[Constants.KEY.MASKED_POS]),
        name=Constants.KEY.MASKED_POS
      )

    self.utt_type = None
    if self.is_multi:
      self.utt_type = tf.placeholder_with_default(
        features[Constants.KEY.UTT_TYPE],
        shape=get_shape_list(features[Constants.KEY.UTT_TYPE]),
        name=Constants.KEY.UTT_TYPE
      )

    # Create masks
    enc_padding_mask, combined_mask, dec_padding_mask, dec_conv_mask = create_masks(encoder_input, decoder_input)

    # Build transformer model
    m_types = Constants.MODEL.get_members()
    if model_type not in m_types:
      first_line = "Wrong model_type: %s" % model_type
      second_line = "Available model_types: %s" % ', '.join(m_types)
      raise ValueError("%s. %s" % (first_line, second_line))

    self.transformer = transformer = self._load_transformer(model_type, enc_len, dec_len)

    if model_type == Constants.MODEL.VANILLA_TRANSFORMER:
      predictions, _ = transformer(
        encoder_input,
        decoder_input,
        training=self.is_training,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask
      )
    else:
      predictions, _ = transformer(
        encoder_input,
        decoder_input,
        training=self.is_training,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask,
        conv_mask=dec_conv_mask,
        use_utt_type=self.is_multi,
        utt_type=self.utt_type
      )

    # Metrics
    self.loss = loss = loss_function(decoder_output, predictions)
    self.probs = predictions
    self.predictions = tf.argmax(predictions, axis=-1, output_type=tf.int64)
    self.accuracy = accuracy = accuracy_function(decoder_output, self.predictions)

    # Make summaries
    summary_accuracy = tf.summary.scalar('accuracy', accuracy)
    summary_loss = tf.summary.scalar('loss', loss)
    self.merged_summary = tf.summary.merge([summary_accuracy, summary_loss])

  def _load_transformer(self, model_type, enc_len, dec_len):
    if model_type == Constants.MODEL.VANILLA_TRANSFORMER:
      transformer = Transformer(
        num_layers=self.params.get("num_layers"),
        d_model=self.params.get("d_model"),
        num_heads=self.params.get("num_heads"),
        dff=self.params.get("dff"),
        vocab_size=self.vocab_size,
        pe_encoder_len=enc_len,
        pe_decoder_len=dec_len,
        rate=self.params.get("dropout_rate"),
        is_embed_pos=self.params.get("is_embed_pos"),
        is_finetune=self.params.get("is_finetune")
      )

    elif model_type == Constants.MODEL.EVOLVED_TRANSFORMER:
      transformer = EvolvedTransformer(
        num_encoder_layers=self.params.get("num_encoder_layers"),
        num_decoder_layers=self.params.get("num_decoder_layers"),
        d_model=self.params.get("d_model"),
        num_heads=self.params.get("num_heads"),
        dff=self.params.get("dff"),
        vocab_size=self.vocab_size,
        type_size=self.params.get('type_size'),
        pe_encoder_len=enc_len,
        pe_decoder_len=dec_len,
        rate=self.params.get("dropout_rate"),
        is_embed_pos=self.params.get("is_embed_pos"),
        is_finetune=self.params.get("is_finetune")
      )

    else:
      transformer = NERTransformer(
        num_encoder_layers=self.params.get("num_encoder_layers"),
        num_decoder_layers=self.params.get("num_decoder_layers"),
        d_model=self.params.get("d_model"),
        num_heads=self.params.get("num_heads"),
        dff=self.params.get("dff"),
        vocab_size=self.vocab_size,
        type_size=self.params.get('type_size'),
        ner_size=Constants.NER_SIZE,
        pe_encoder_len=enc_len,
        pe_decoder_len=dec_len,
        rate=self.params.get("dropout_rate"),
        is_embed_pos=self.params.get("is_embed_pos"),
        is_finetune=self.params.get("is_finetune")
      )

    return transformer

  def _restore_checkpoint(self):
    """restore and initialize"""
    initialized_variable_names = {}
    trainable_vars = tf.trainable_variables()
    if self.flag_objs.init_checkpoint:
      if not self.flag_objs.checkpoint_type:
        raise ValueError("If 'init_checkpoint' is not None, 'checkpoint_type' must be also set.")

      elif self.flag_objs.checkpoint_type == "p":
        if self.verbose:
          print("Pretrain checkpoint will be restored.")
        checkpoint_dir = os.path.join(
          Constants.CHECKPOINT_DIR, Constants.PRETRAIN_CKP, self.flag_objs.ckp_workspace
        )

      elif self.flag_objs.checkpoint_type == "f":
        if self.verbose:
          print("Finetune checkpoint will be restored.")
        checkpoint_dir = os.path.join(
          Constants.CHECKPOINT_DIR, Constants.FINETUNE_CKP, self.flag_objs.ckp_workspace
        )

      else:
        raise ValueError("Not defined 'checkpoint_type' : %s" % self.flag_objs.checkpoint_type)

      init_ck = handle_init_checkpoint(checkpoint_dir, self.flag_objs.init_checkpoint)

      if self.is_saver_to_load:
        saver = tf.train.Saver()
        saver.restore(self.sess, init_ck)
        return

      assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
        trainable_vars, init_ck)
      tf.train.init_from_checkpoint(init_ck, assignment_map)

    # List up trainable variables
    if self.verbose:
      print("**** Trainable Variables ****")
      for var in trainable_vars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    self.sess.run(tf.global_variables_initializer())
    return

  def fit(self, mode):
    """train model with session"""
    is_pretrain = mode in ['p', 'pretrain']
    global_step = tf.train.get_or_create_global_step()

    # Get training ops
    train_op = train_optimizer(
      loss=self.loss,
      init_learning_rate=self.params.get("learning_rate"),
      num_train_steps=self.params.get("num_train_steps"),
      num_warmup_steps=self.params.get("num_warmup_steps"),
      global_step=global_step,
      use_lamb=self.params.get("use_lamb", False)
    )

    # Make Checkpoint dir
    save_dir = Constants.PRETRAIN_CKP if is_pretrain else Constants.FINETUNE_CKP
    checkpoint_dir = os.path.join(
      Constants.CHECKPOINT_DIR, save_dir, self.flag_objs.workspace
    )
    epoch_ck_path = os.path.join(checkpoint_dir, "model")
    summary_path = os.path.join(checkpoint_dir, 'summaries')
    tf.gfile.MakeDirs(checkpoint_dir)

    # Load ckpt saver
    max_to_keep = Constants.MAX_TO_KEEP
    train_summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
    epoch_saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
    if self.verbose:
      print("Saver max_to_keep: %d" % max_to_keep)

    # Initialize model
    self._restore_checkpoint()
    if self.is_saver_to_load and not is_pretrain and (self.flag_objs.checkpoint_type == "p"):
      self.sess.run(tf.variables_initializer([global_step]))
    if self.verbose:
      print("Initialize models")

    _step = 0
    for cur_epoch in range(self.params.get("num_epochs")):
      if self.verbose:
        print("New epoch start : [%d]" % cur_epoch)

      # Initialize dataset iterator
      self.sess.run(self.input_initializer)
      if self.verbose:
        print("Dataset is initialized with train tfrecord")

      while True:
        # Stop training if step is exceed 'num_train_steps'
        if _step >= self.params.get("num_train_steps"):
          break

        try:
          _masked_pos = None
          if is_pretrain:
            (_train, _loss, _accuracy, _predictions, _step, _merge,
             _enc_in, _dec_out, _masked_pos) = self.sess.run([
              train_op, self.loss, self.accuracy, self.predictions, global_step, self.merged_summary,
              self.encoder_input, self.decoder_output, self.masked_position])
          else:
            (_train, _loss, _accuracy, _predictions, _step, _merge,
             _enc_in, _dec_out) = self.sess.run([
              train_op, self.loss, self.accuracy, self.predictions, global_step, self.merged_summary,
              self.encoder_input, self.decoder_output])

          train_summary_writer.add_summary(_merge, _step)

          # Print training information
          if self.verbose and (_step % Constants.PRINT_CYCLE_STEP == 0):
            print("[ %d ] loss: %.5f, acc: %.2f%%" % (_step, _loss, _accuracy))
            inputs = [_enc_in[0], _dec_out[0], _predictions[0]]
            if is_pretrain:
              inputs.append(_masked_pos[0])
            self._print_sample(inputs, is_pretrain)

          if _step % Constants.SAVE_CYCLE_STEP == 0:
            epoch_saver.save(self.sess, epoch_ck_path, global_step=_step)

        except tf.errors.OutOfRangeError:
          # Save checkpoint and hyper parameter info
          # epoch_saver.save(self.sess, epoch_ck_path, global_step=_step)
          break

    # Save final checkpoint
    epoch_saver.save(self.sess, epoch_ck_path, global_step=_step)
    return

  def _print_sample(self, inputs, is_pretrain):
    """"""
    if is_pretrain:
      ex_enc_in, ex_dec_out, ex_pred, ex_mask_pos = inputs

      # Remove pad
      length = np.count_nonzero(ex_mask_pos)
      ex_dec_out = ex_dec_out[:length]
      ex_pred = ex_pred[:length]
      ex_mask_pos = ex_mask_pos[:length]

      # Fill masked part
      real, pred = np.copy(ex_enc_in), np.copy(ex_enc_in)
      real[ex_mask_pos] = ex_dec_out
      pred[ex_mask_pos] = ex_pred

      #
      print("  <sentence>")
      print("    - original  : %s" % self.tokenizer.decode(real.tolist()))
      print("    - prediction: %s" % self.tokenizer.decode(pred.tolist()))
      print("  <token>")
      print("    - original  : %s" % self.tokenizer.decode_by_token(ex_dec_out.tolist()))
      print("    - prediction: %s" % self.tokenizer.decode_by_token(ex_pred.tolist()))
      return

    # else:
    ex_enc_in = inputs[0].tolist()
    ex_dec_out = inputs[1].tolist()
    ex_pred = inputs[2].tolist()

    #
    text_q = self.tokenizer.decode(ex_enc_in)
    if self.tokenizer.get_eos_id() in ex_pred:
      ex_pred = ex_pred[:ex_pred.index(self.tokenizer.get_eos_id())]

    #
    print("  [ Q       ] %s" % text_q)
    print("  [ A - real] %s" % self.tokenizer.decode(ex_dec_out))
    print("  [ A - pred] %s" % self.tokenizer.decode(ex_pred))
    print("  [ A - gen ] %s" % self.generate(text_q, do_sample=self.flag_objs.do_sample))
    return

  def generate(self, q_text, a_text='', do_sample=True, top_k=50):
    """
    generate text with top-k sampling
    :param q_text: 'question' or context=['c1', 'c2', ...]
    :param a_text: 'prev tokens for answer'
    :param do_sample: top_k sampling if True else greedy search
    :param top_k: the number of top k tokens to use
    :return:
    """
    eos_id = self.tokenizer.get_eos_id()
    seq_len_decoder = self.flag_objs.seq_len_decoder
    batch_size = self.flag_objs.batch_size

    enc_in, dec_in, _, _, utt_type, len_dec_in = self.data_handler.transform_text_to_input_feature(
      a_text, context_list=q_text, ner_texts_list=None, do=True)
    # enc_in, dec_in, len_dec_in = self.preprocess(q_text, a_text)  # TODO: erase (EX9 때문에 남겨둠)
    dec_out = []

    while len_dec_in < seq_len_decoder:
      feed_dict = {
        self.encoder_input: [enc_in] * batch_size,
        self.decoder_input: [dec_in] * batch_size
      }
      if self.is_multi:
        feed_dict[self.utt_type] = [utt_type] * batch_size

      # (batch_size, seq_len, vocab_size) --> (vocab_size)
      probs = self.sess.run(self.probs, feed_dict=feed_dict)[0][len_dec_in-1]
      pred_token = int(np.argmax(probs))    # np.int64 --> int
      if do_sample:
        # probs 중에서 확률이 높은 top_k를 뽑고 probs 에 기반하여 sampling
        sample = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:top_k]
        _ind, _prob = list(zip(*sample))
        _prob = [x / sum(_prob) for x in _prob]
        pred_token = int(np.random.choice(_ind, 1, p=_prob)[0])

      dec_out.append(pred_token)
      dec_in[len_dec_in] = pred_token
      len_dec_in += 1

      if pred_token == eos_id:
        break

    return self.tokenizer.decode(dec_out)

  def variable_checker(self, do_initialize=False):
    """Initialized uninitialized variables"""
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = self.sess.run(
      [~(tf.is_variable_initialized(var)) for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))
    if len(not_initialized_vars):
      if self.verbose:
        print("#Global_variables: %d / #uninitialized_variables: %d)" % (
          len(global_vars), len(not_initialized_vars)))
      if do_initialize:
        self.sess.run(tf.variables_initializer(not_initialized_vars))

  def preprocess(self, q_text, a_text, is_bos=True):
    """
    TODO: EX9 때문에 살려놨는데 나중에 지울 것
    :param q_text: 'question'
    :param a_text: 'answer'
    :param is_bos: if true, add [BOS] and [EOS] to encoder_input
    :return:
    """
    bos_id = self.tokenizer.get_bos_id()
    eos_id = self.tokenizer.get_eos_id()
    pad_id = self.tokenizer.get_pad_id()

    enc_target_length = self.flag_objs.seq_len_encoder
    dec_target_length = self.flag_objs.seq_len_decoder
    if is_bos:
      enc_target_length -= 2
    dec_target_length -= 1

    #
    q_text, a_text = q_text.strip(), a_text.strip()
    q_out = self.tokenizer.encode(q_text)
    a_out = self.tokenizer.encode(a_text)

    # max_seq_len 넘는 문장의 경우 앞 토큰 제거
    seq_len_q, seq_len_a = len(q_out), len(a_out)
    if seq_len_q > enc_target_length:
      q_out = q_out[-enc_target_length:]
    if seq_len_a > dec_target_length:
      a_out = a_out[-dec_target_length:]

    # BOS, EOS 추가
    if is_bos:
      q_out = [bos_id] + q_out + [eos_id]

    # ENCODER_INPUT, DECODER_INPUT
    enc_in = q_out
    dec_in = [bos_id] + a_out
    len_dec_in = len(dec_in)

    # PADDING
    pad_len_enc = self.flag_objs.seq_len_encoder - len(enc_in)
    pad_len_dec = self.flag_objs.seq_len_decoder - len_dec_in
    enc_in += [pad_id] * pad_len_enc
    dec_in += [pad_id] * pad_len_dec
    return enc_in, dec_in, len_dec_in
