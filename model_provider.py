from nameko.extensions import DependencyProvider
import tensorflow as tf
from deeppavlov.agents.paraphraser.paraphraser import EnsembleParaphraserAgent
import os
import json


class Agent(DependencyProvider):
    def __init__(self):
        config_path = os.path.join(os.environ.setdefault('PARAPHRASER_DATA', '.'), "config.json")
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        opt = config["opt"]
        self.agent = EnsembleParaphraserAgent(opt)
        self.graph = tf.get_default_graph()

    def get_dependency(self, worker_ctx):
        return {
            'agent': self.agent,
            'graph': self.graph
        }
