
with open('../tokenizer/2020-06-18_wiki_30000.vocab') as fp:
  lines = fp.readlines()

with open('../tokenizer/single_tok.txt') as fp:
  lines_origin = fp.readlines()

# print(lines)
v_list = []
s_list = []
for line in lines:
  v, s = line.split()
  v_list.append(v.strip())
  s_list.append(s)


cnt = 0
for v_origin in lines_origin:
  v_origin = v_origin.strip()
  if v_origin not in v_list:
    print("v_origin: %s" % v_origin)
    cnt += 1

print("cnt: %d" % cnt)

