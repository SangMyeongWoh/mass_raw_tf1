from elasticsearch import Elasticsearch
from pathlib import Path
import os
import json
import sys

es = Elasticsearch("192.168.2.75:9200")
dir = './from_elastic/naver'
index = "navernews_crawling"
body = {
  'query': {
    "match":{
      "_id": "1_social_c_0002474399"
    }
  }
}


result = es.search(
  index=index,
  scroll='3s',
  size=100,
  body=body
)

print((result['hits']['hits']))


# iteration_num = 1
#
# file_index = 1
# file_index_now = 0
#
# for val in result['hits']['hits']:
#     print(val['_source']['text'] + '\n')
#     underbar_count = 0
#     for ch_index in range(len(val['_id'])):
#       if val['_id'][ch_index] == "_":
#         underbar_count = underbar_count + 1
#       if underbar_count == 4:
#         print(val['_id'][:ch_index])








