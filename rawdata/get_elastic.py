from elasticsearch import Elasticsearch
from pathlib import Path
import os

def get_news_id(id):
  underbar_count = 0
  for ch_index in range(len(id)):
    if id[ch_index] == "_":
      underbar_count = underbar_count + 1
    if underbar_count == 4:
      return id[:ch_index]

es = Elasticsearch("192.168.2.75:9200", timeout=30)
dir_comment = './from_elastic/naver_comment'
dir_news = './from_elastic/naver_news'
index = "navernews_crawling"
body = {
  "query": {
        "bool": {
            "must_not": {
                "exists": {
                    "field": "parent_id"
                }
            }
        }
    }
}

result = es.search(
  index=index,
  scroll='25m',
  size=500,
  body=body
)

sid = result['_scroll_id']
iteration_num = 1

file_index_comment = 1
file_index_now_comment = 1

file_index_news = 1
file_index_now_news = 1

news_id_now = ''

filename_comment = 'naver_comment_'
filename_news = 'naver_news_'

#fw = open(os.path.join(dir_comment, filename_comment + str(file_index_comment) + '.txt'), 'w')
fw2 = open(os.path.join(dir_news, filename_news + str(file_index_news) + '.txt'), 'w')
fw3 = open('./log.txt', 'w')


while(len(result['hits']['hits']) > 0):

  # """news below"""
  #
  # if file_index_news != file_index_now_news:
  #   file_index_now_news = file_index_news
  #   fw2 = open(os.path.join(dir_news, 'naver_news_' + str(file_index_news) + '.txt'), 'w')


  result = es.scroll(scroll_id=sid, scroll='25m')
  iteration_num = iteration_num + 1
  print(iteration_num)
  try:
    for val in result['hits']['hits']:
      """comment below"""

      if file_index_comment != file_index_now_comment:
        fw.close()
        print("file created_comment")
        file_index_now_comment = file_index_comment
        fw = open(os.path.join(dir_comment, filename_comment + str(file_index_comment) + '.txt'), 'w')

      if file_index_news != file_index_now_news:
        fw2.close()
        print("file created_news")
        file_index_now_news = file_index_news
        fw2 = open(os.path.join(dir_news, filename_news + str(file_index_news) + '.txt'), 'w')


      if 'parent_id' in val['_source'].keys():
        fw.write(val['_source']['text'] + '\n')
      else:
        fw2.write(val['_source']['text'])




      # if Path(os.path.join(dir_comment, filename_comment + str(file_index_comment) + '.txt')).stat().st_size > 1000000000:
      #   file_index_comment = file_index_comment + 1
      if Path(os.path.join(dir_news, filename_news + str(file_index_news) + '.txt')).stat().st_size > 1000000000:
        file_index_news = file_index_news + 1

  except:
    fw3.write(str(iteration_num) + '\n')
    print("error accur")







