from nameko.extensions import DependencyProvider
import tensorflow as tf
from sklearn.externals import joblib
from deeppavlov.agents.paraphraser.paraphraser import EnsembleParaphraserAgent
import os
import json


class Agent(DependencyProvider):
    agent = None
    graph = None

    def __init__(self):
        if Agent.agent is None:
            print("Init Agent")
            config_path = os.path.join(os.environ.setdefault('PARAPHRASER_DATA', '.'), "config.json")
            with open(config_path, "r") as f:
                config = json.loads(f.read())
            opt = config["opt"]
            Agent.agent = EnsembleParaphraserAgent(opt)
            Agent.graph = tf.get_default_graph()
        else:
            print("Agent already initialized")

    def get_dependency(self, worker_ctx):
        return {
            'agent': Agent.agent,
            'graph': Agent.graph
        }


class Classifier(DependencyProvider):
    def __init__(self):
        config_path = os.path.join(os.environ.setdefault('PARAPHRASER_DATA', '.'), "config.json")
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        self.classifier = joblib.load(config["classifier"])

    def get_dependency(self, worker_ctx):
        return {
            'classifier': self.classifier
        }

