import re
from os import walk

except_val = ["기자", "URL", "/", "[", "]", "<", ">", "(", ")", "	", "com"]

files = []
for (dirpath, dirnames, filenames) in walk("./before"):
    files.extend(filenames)

with open('news.txt', 'w') as outfile:
  for fname in files:
    with open("./before/" + fname) as infile:
      outfile.write(infile.read())

with open('news_preprocessed.txt', 'w') as outfile:
  with open('news.txt', 'r') as inputfile:
    news = inputfile.readlines()
    for line in news:
      index = 0
      for ch_index in range(len(line)):
        if line[ch_index] == '.':
          new_line = line[index:ch_index]

          index = ch_index + 1
          if len(new_line) < 30:
            continue
          if any(x in new_line for x in except_val):
            continue
          new_line = new_line.strip()
          outfile.write(new_line + "\n")




    # print(line)
    # if line == "\n":
    #   continue
    # if 'title' in line:
    #   continue
    # if len(line) < 8:
    #   continue
    # if '@' in line:
    #   continue
    # if '————' in line:
    #   continue
    # line = line.strip()
    # line = line.replace('-', '')

