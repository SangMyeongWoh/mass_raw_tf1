class APIManager(object):

  def __init__(self):
    self.chat_ids = list()
    self.conversations = dict()

  def generate_key(self):
    chat_id = 0 if len(self.chat_ids) < 1 else max(self.chat_ids) + 1
    self.chat_ids.append(chat_id)
    self.conversations[chat_id] = list()
    return dict(chat_id=chat_id)

  def _confirm_chat_id(self, chat_id):
    if chat_id not in self.conversations:
      id_lists = ', '.join(map(str, self.conversations.keys()))
      raise ValueError("Wrong chat_id '%s'. Available chat_ids: %s." % (chat_id, id_lists))
    return

  def update_conversation(self, chat_id, utterance):
    self._confirm_chat_id(chat_id)
    self.conversations[chat_id].append(utterance)
    return

  def get_conversation(self, chat_id, max_context=7):
    self._confirm_chat_id(chat_id)
    return self.conversations[chat_id][-max_context:]