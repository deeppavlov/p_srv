from nameko.extensions import DependencyProvider
import tensorflow as tf
from deeppavlov.agents.paraphraser.paraphraser import EnsembleParaphraserAgent
import os


class Agent(DependencyProvider):
    def __init__(self):
        data_path = os.environ['PARAPHRASER_DATA']
        opt = {
            'fasttext_model': os.path.join(data_path, 'fasttext.bin'),
            'model_files': [os.path.join(data_path, 'models/maxpool_match_%i' % i) for i in range(5)],
            'datatype': 'test'

        }
        self.agent = EnsembleParaphraserAgent(opt)
        self.graph = tf.get_default_graph()

    def get_dependency(self, worker_ctx):
        return {
            'agent': self.agent,
            'graph': self.graph
        }
