from nameko.extensions import DependencyProvider
import tensorflow as tf
from deeppavlov.agents.paraphraser.paraphraser import ParaphraserAgent
import os

class Agent(DependencyProvider):
    def __init__(self):
        data_path = os.environ['PARAPHRASER_DATA']
        opt = {
            'fasttext_model': os.path.join(data_path, 'fasttext.bin'),
            'pretrained_model': os.path.join(data_path, 'paraphraser')
        }
        self.agent = ParaphraserAgent(opt)
        # self.model = ParaphraserModel(opt)
        self.graph = tf.get_default_graph()

    def get_dependency(self, worker_ctx):
        return {
            'agent': self.agent,
            'graph': self.graph
        }
