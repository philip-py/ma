from flask import Flask, jsonify
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from app.config import config
from logzero import setup_logger

bootstrap = Bootstrap()
db = SQLAlchemy()

def create_app(config_name):
    app = Flask(__name__)

    # See http://flask.pocoo.org/docs/latest/config/
    app.config.from_object(config[config_name])
    app.config.update(dict(DEBUG=True))

    config[config_name].init_app(app)
    bootstrap.init_app(app)
    db.init_app(app)

    logger = setup_logger(app.config.get('LOGFILE'), disableStderrLogger=True)

    @app.route("/")
    def hello_world():
        logger.info("/")
        return "Hello World"

    @app.route("/foo/<someId>")
    def foo_url_arg(someId):
        logger.info("/foo/%s", someId)
        return jsonify({"echo": someId})

    return app
