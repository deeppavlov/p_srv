from nameko.rpc import rpc
import random
import json
from embeddings_dict import EmbeddingsDict
from model import ParaphraserModel


class Paraphraser(object):
    name = 'paraphraser'

    @rpc
    def predict(self, phrase=None, phrases=None):
        opt = dict()
        opt['datapath'] = '/home/aleksandr/Dev/data'
        opt['fasttext_model'] = 'model_yalen_sg_300.bin'
        opt['fasttext_dir'] = '/home/aleksandr/Dev/fastText'
        opt['pretrained_model'] = '/home/aleksandr/Dev/data/paraphrases/paraphraser'

        embdict = EmbeddingsDict(opt)
        model = ParaphraserModel(opt, embdict)
        single = [{
            'text': 'Общая фраза\nСколько стоит чайник?\nПо чем кофеварка?'
        }]
        batch = model.batchify([model.build_ex(ex) for ex in single])

        prediction = model.predict(batch)
        return [prediction]

if __name__ == '__main__':
    p = Paraphraser()
    print(p.predict())
