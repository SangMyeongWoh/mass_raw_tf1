from absl import flags
from flask import Flask
from flask_cors import CORS
from flask_restful_swagger_2 import Api

from configs.constants import Constants
from apis.amumal import amumal_blueprint


flag_objs = flags.FLAGS
app = Flask(__name__)

# setting CORS
cors = CORS(app, resources={r"*": {"origins": "*"}})

# setting swagger api doc
api = Api(app, title='API Template', api_version='0.0.1', api_spec_url='/swagger',
          host='localhost', description='API Template')

# add blueprint. URL 에는 Dash(-) 를 사용하는 것이 추천된다.
app.register_blueprint(amumal_blueprint, url_prefix=Constants.API.PREFIX_AMUMAL)


@app.before_request
def before_request():
  """
  If an API called, below precess is called.

  <app before request>
  <blueprint before request>
  <blueprint called api task>
  <blueprint after request>
  <app after request>

  :return:
  """
  pass


@app.after_request
def after_request(response):
  return response


def run_api():
  app.run(
    host=flag_objs.host,
    port=flag_objs.port,
    threaded=True,
    debug=False
  )