import os
import sys

sys.path.append('..')

from src.data_handler import join_conversation

PATH = '../data/conversation'

total_dialogs, total_utterance, total_char = 0, 0, 0
for file_name in os.listdir(PATH):
  # IGNORE CSV FILE
  if file_name.endswith('.csv'):
    continue

  # LOAD
  path_to_file = os.path.join(PATH, file_name)
  dialogs = join_conversation(path_to_file=path_to_file)

  # PRINT STATISTICS
  dialog_length = [len(x) for x in dialogs]
  cnt_dialog = len(dialogs)
  cnt_utterance = sum(dialog_length)
  print('[STATISTICS: %s]' % file_name)
  print('  - the number of dialogs:', f"{cnt_dialog:,}")
  print('  - the number of utterances:', f"{cnt_utterance:,}")
  print('    # the mean number of utterance per dialogs: %.1f' % (cnt_utterance / cnt_dialog))
  print('    # the min number of utterance per dialogs: %d' % min(dialog_length))
  print('    # the max number of utterance per dialogs: %d' % max(dialog_length))

  # TOTAL STATS
  total_dialogs += cnt_dialog
  total_utterance += cnt_utterance
  total_char += sum([len(x) for y in dialogs for x in y])

print('[TOTAL]')
print('  - the number of dialogs:', f"{total_dialogs:,}")
print('  - the number of utterances:', f"{total_utterance:,}")
print('  - the number of characters:', f"{total_char:,}")
