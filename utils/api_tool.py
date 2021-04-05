import json
from flask import make_response


def return_error(error_msg, error_code=400):
  ret = dict(error_code=error_code, error_msg=error_msg)
  return wrap_output(ret, error_code)


def wrap_output(output_data, http_return_code=200):
  output = make_response(json.dumps(output_data, ensure_ascii=False), http_return_code)
  output.headers['Cache-Control'] = 'no-cache,no-store,must-revalidate'
  output.headers.extend({
    "content-type": "application/json"
  })
  return output