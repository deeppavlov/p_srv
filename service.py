from nameko.rpc import rpc
from keras.models import Model
import numpy as np
from model_provider import Agent, Classifier
from faq_provider import Faq


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


class FaqParaphraser(object):

    name = 'faq_paraphraser'

    a = Agent()
    c = Classifier()
    faq = Faq()

    @rpc
    def predict(self, phrase=None):
        with self.a['graph'].as_default():
            if phrase is None:
                return []
            phrase_vector = self.__get_vectors([phrase])
            prediction = self.c["classifier"].predict(phrase_vector)
            proba = self.c["classifier"].predict_proba(phrase_vector)
            max_prob = proba[0, np.argmax(proba, axis=1)]
            return self.faq["questions_groups"][prediction[0]][0], self.faq["answers"][prediction[0]], max_prob[0]

    def __get_inputs(self, phrase=None, phrases=None):
        if phrases is None or len(phrases) == 0:
            return []
        observations = []
        for p in phrases:
            observations.append({
                'id': 'p_srv',
                'text': 'Dummy title\n%s\n%s' % (phrase, p)
            })
        examples = [self.a["agent"].models[0].build_ex(obs) for obs in observations]
        examples = [ex for ex in examples if ex is not None]
        batch = self.a["agent"].models[0].batchify(examples)
        batch, _ = batch
        return batch

    def __get_vectors_from_model(self, sen_li, ind):
        data = self.__get_inputs(phrases=sen_li)
        model = self.a["agent"].models[ind].model
        layer_name = 'biLSTM_encoder_last'
        layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).get_output_at(2))
        output = layer_model.predict(data)
        return output

    def __get_vectors(self, sentences_list):
        vectors_list = []
        for i in range(len(self.a["agent"].models)):
            vectors_list.append(self.__get_vectors_from_model(sentences_list, i))
        vectors_array = np.mean(np.array(vectors_list), axis=0, keepdims=False)
        return vectors_array
