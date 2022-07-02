import os
import json
from flask import Flask, jsonify, request

from data import all_categories, all_models, get_category, get_model, update_model
from util import test
app = Flask(__name__)


@app.route("/")
def route_hello_world():
    return "Hello, World!"


@app.route("/models")
def route_all_models():
    return jsonify(all_models())


@app.route("/models/<id>", methods=["GET"])
def route_get_model(id: str):
    return jsonify(get_model(int(id)))


@app.route("/models/<id>", methods=["POST"])
def route_update_model(id: str):
    data = request.get_json()
    return jsonify(update_model(int(id), data))


@app.route("/categories")
def route_all_categories():
    return jsonify(all_categories())


@app.route("/categories/<id>")
def route_get_category(id: str):
    return jsonify(get_category(int(id)))


if __name__ == "__main__":
    """Entry point for REST API mockup
    """
    #path = os.getcwd()
    # dir = os.path.dirname(path) #get parent folder
    # path = "../storage/config.json" #if sibling folder required
    #print(path)
    root_dr = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..\..')
    path = os.path.join(root_dr, "storage", "config.json")
    config = json.load(open(path, 'r'))

    """Load all settings from config"""
    outcomes_dict = config["outcomes"]
    outcomes_matrix_dict = config["outcomes_matrix"]
    exclusions_dict = config["exclusions"]
    features_dict = config["features"]
    events_dict = config["events"]
    models_dict = config["models"]
    print(models_dict)
    app.run(debug=True)
