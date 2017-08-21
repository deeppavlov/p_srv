from nameko.rpc import rpc
import random
import json
from embeddings_dict import EmbeddingsDict
from model import ParaphraserModel


class Paraphraser(object):
    name = 'paraphraser'

    def __init__(self):
        opt = dict()
        opt['datapath'] = '/home/aleksandr/Dev/data'
        opt['fasttext_model'] = 'model_yalen_sg_300.bin'
        opt['fasttext_dir'] = '/home/aleksandr/Dev/fastText'
        opt['pretrained_model'] = '/home/aleksandr/Dev/data/paraphrases/paraphraser'
        opt['embedding_dim'] = 300

        self.embdict = EmbeddingsDict(opt)
        self.model = ParaphraserModel(opt, self.embdict)

    @rpc
    def predict(self, phrase=None, phrases=[]):
        data = []
        for p in phrases:
            data.append({
                'text': 'Dummy title\n%s\n%s' % (phrase, p)
            })
        batch, _ = self.model.batchify([self.model.build_ex(ex) for ex in data])
        prediction = self.model.predict(batch)
        return prediction

if __name__ == '__main__':
    p = Paraphraser()
    print(p.predict("Сколько стоит чайник", ["По чем кофеварка", "Сколько стоит пылесос", "По чем чай"]))
