import csv
import string


f = open('petition.txt', 'w')
with open('petition.csv', 'r') as csvfile:
  #reader = csv.reader(csvfile, delimiter=' ')
  for row in csvfile:
    row = row.split(",", 2)
    if "------" in row[2]:
      print(row[2])
      continue
    if "=====" in row[2]:
      print(row[2])
      continue

    row[2] = row[2].replace('\\n', '')
    index = 4
    startindex = 0
    for i in range(len(row[2])):
      if row[2][i] == ',':
        index = index - 1
        startindex = i
        if index == 0:
          break
    print(row[2][startindex + 1:])
    f.write(row[2][startindex + 1:])
    #print(row[2][15:])
    print("\n\n")

f.close()