from nameko.rpc import rpc
from model_provider import Agent


class Paraphraser(object):

    name = 'paraphraser'

    a = Agent()

    @rpc
    def predict(self, phrase=None, phrases=[]):
        if phrases is None or len(phrases)==0:
            return []
        with self.a['graph'].as_default():
            observations = []
            for p in phrases:
                observations.append({
                    'id': 'p_srv',
                    'text': 'Dummy title\n%s\n%s' % (phrase, p)
                })
            predictions = self.a['agent'].batch_act(observations)
            result = []
            for p in predictions:
                score = p['score'].tolist()[0]
                result.append(score)
            print(result)
            return result
