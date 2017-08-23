from nameko.rpc import rpc
from model_provider import Model


class Paraphraser(object):

    name = 'paraphraser'

    m = Model()

    @rpc
    def predict(self, phrase=None, phrases=[]):
        if phrases is None or len(phrases)==0:
            return []
        with self.m['graph'].as_default():
            data = []
            for p in phrases:
                data.append({
                    'text': 'Dummy title\n%s\n%s' % (phrase, p)
                })
            batch, _ = self.m['model'].batchify([self.m['model'].build_ex(ex) for ex in data])
            prediction = self.m['model'].predict(batch)
            result = prediction.tolist()
            print(result)
            return result
