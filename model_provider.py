from nameko.extensions import DependencyProvider
import tensorflow as tf

from model.model import ParaphraserModel
from model.embeddings_dict import EmbeddingsDict


class Model(DependencyProvider):
    def __init__(self):
        opt = dict()
        opt['datapath'] = '/home/aleksandr/Dev/data'
        opt['fasttext_model'] = 'model_yalen_sg_300.bin'
        opt['fasttext_dir'] = '/home/aleksandr/Dev/fastText'
        opt['pretrained_model'] = '/home/aleksandr/Dev/data/paraphrases/paraphraser'
        opt['embedding_dim'] = 300
        self.model = ParaphraserModel(opt, EmbeddingsDict(opt))
        self.graph = tf.get_default_graph()

    def get_dependency(self, worker_ctx):
        return {
            'model': self.model,
            'graph': self.graph
        }
