from nameko.extensions import DependencyProvider
import os
import json
import csv


class Faq(DependencyProvider):
    def __init__(self):
        config_path = os.path.join(os.environ.setdefault('PARAPHRASER_DATA', '.'), "config.json")
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        faq = config["faq"]
        self.questions_groups, self.answers = self.__get_faq(faq)

    def get_dependency(self, worker_ctx):
        return {
            'questions_groups': self.questions_groups,
            'answers': self.answers
        }

    @staticmethod
    def __get_faq(faq=None):
        with open(faq, 'r') as csv_file:
            lines = csv.reader(csv_file, delimiter='\n')
            strings = []
            for row in lines:
                strings.append(row[0])
            csv_file.close()
            strings = strings[1:]
            questions_strings = [el.split('\t')[0] for el in strings]
            answers = [el.split('\t')[1] for el in strings]
            questions_groups = [el.split('#') for el in questions_strings]
            return questions_groups, answers
