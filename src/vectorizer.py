import gensim
import json


class Vectorizer:
    def __init__(self, model: gensim.models.fasttext.FastTextKeyedVectors) -> None:
        self.model = model
        self._vectors_dictionary = {'BOS': self.model['BOS'].tolist(),
                                    'EOS': self.model['EOS'].tolist(),
                                    'PEOS': self.model['PEOS'].tolist()}

    def update_dict(self, words: str) -> None:
        """
        Updating the dictionary during each cell vectorising
        """
        for one_word in words.split(', '):
            if one_word not in self._vectors_dictionary:
                self._vectors_dictionary[one_word] = self.model[one_word].tolist()

    def update_json(self) -> None:
        """
        Updating and saving the json file
        """
        with open("/content/vectors.json", "w") as fp:
            json.dump(self._vectors_dictionary, fp, ensure_ascii=False)

    def get_dictionary(self) -> dict:
        """
        In case we need to get the dictionary
        """
        return self._vectors_dictionary

    @staticmethod
    def get_sequence(words_string: str) -> list[str]:
        """
        Getting a list of tokens + tags of beginning and ending
        BOS -- Beginning of Sentence
        PEOS -- pre-End of Sentence
        EOS -- End of Sentence
        """
        return ['BOS'] + words_string.split(', ') + ['PEOS', 'EOS']

