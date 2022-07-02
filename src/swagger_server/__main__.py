import logging
import logging.handlers
from cgitb import handler
import time
import click
import connexion
#from typing import List, Dict  # noqa: F401
from flask import current_app
from swagger_server import encoder
from swagger_server.util import get_config_data


@click.command()
@click.option('-p', '--port', type=click.INT, required=False, default=5000, help="Service port")
@click.option('-c', '--config', type=click.STRING, required=False, default="config.json", help="config path json|yaml")
@click.option('-l', '--loglevel', type=click.STRING, required=False, default="WARNING", help="logging levels: CRITICAL|ERROR|WARNING|INFO|DEBUG|NOTSET")
def main(port, config, loglevel):
    # Logging. By default, the log file grows indefinitely.
    # logging.basicConfig(filename='./api.log', encoding='utf-8', level=loglevel)

    # If we need rotating logs like: app.log, app.log.1, app.log.2 ... app.log.{backupCount}
    # https://docs.python.org/3/library/logging.handlers.html#filehandler
    handlers = [
        # logging.FileHandler('./api.log'),  # Default mode='a', encoding=None
        logging.handlers.RotatingFileHandler(
            './api.log', maxBytes=1*1024*1024, backupCount=3),
        logging.StreamHandler(),  # Default stream=sys.stderr
    ]
    logging.Formatter.converter = time.gmtime  # UTC asctime
    logging.basicConfig(handlers=handlers, encoding='utf-8', level=loglevel,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    # app.app.env = "development" #development|production
    app.add_api('swagger.yaml', arguments={
                'title': 'Containerization of machine learning'})
    models, features = get_config_data(config)

    with app.app.app_context():
        current_app.models = models
        current_app.features = features
        current_app.files = {}  # lazy load when used
        current_app.VERSION = "4.0.2"

    click.echo(click.style(
        f'service_port: {port}, config_path: {config}, logging level: {loglevel}', fg='yellow'))
    click.echo(click.style(
        f'{models.__len__()} model(s) are cached by application', fg='green'))
    app.run(host="0.0.0.0", port=port)


if __name__ == '__main__':
    main()
