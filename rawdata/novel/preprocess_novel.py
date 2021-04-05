from os import walk

files = []
for (dirpath, dirnames, filenames) in walk("./before"):
    files.extend(filenames)

print(files)

with open('novel.txt', 'w') as outfile:
    for fname in files:
        with open("./before/" + fname) as infile:
            outfile.write(infile.read())

with open('novel_preprocessed.txt', 'w') as outfile:
  with open('novel.txt', 'r') as inputfile:
    novel = inputfile.readlines()
    for line in novel:
      if line == "\n":
        continue
      if 'title' in line:
        continue
      if len(line) < 8:
        continue
      if '@' in line:
        continue
      if '————' in line:
        continue
      line = line.strip()
      line = line.replace('-', '')
      outfile.write(line + "\n")
