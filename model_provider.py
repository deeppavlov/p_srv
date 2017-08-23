from nameko.extensions import DependencyProvider
import tensorflow as tf

from model.model import ParaphraserModel
from model.embeddings_dict import EmbeddingsDict


class Model(DependencyProvider):
    def __init__(self):
        opt = {
            'fasttext_model': 'ft_0.8.3_yalen_sg_300.bin',
            'pretrained_model': './data/paraphraser'
        }
        self.model = ParaphraserModel(opt)
        self.graph = tf.get_default_graph()

    def get_dependency(self, worker_ctx):
        return {
            'model': self.model,
            'graph': self.graph
        }
