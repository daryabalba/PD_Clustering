import nltk
from nltk import word_tokenize, bigrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import pandas as pd


nltk.download('punkt')

class CollocationFinder:
    def __init__(self):
        pass

    def get_collocations(self, text, word, top_n=20):
        tokens = word_tokenize(text.lower(), language='russian')
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)

        finder.apply_ngram_filter(lambda w1, w2: word not in (w1, w2))
        scored_collocations = finder.score_ngrams(bigram_measures.student_t)

        return scored_collocations[:top_n]

    def process_texts(self, texts, words, top_n=20, text_processor):
        combined_text = ' '.join(texts)
       
        raw_collocations = {word: self.get_collocations(combined_text, word, top_n) for word in words}

        lemmatized_text = ' '.join(self.text_processor.lemmatize_text(combined_text))

        lemmatized_collocations = {word: self.get_collocations(lemmatized_text, word, top_n) for word in words}

        return raw_collocations, lemmatized_collocations

    def save_to_excel(self, words, raw_collocations, lemmatized_collocations, file_name='collocations.xlsx'):
        rows = []

 
        for word in words:
            for i in range(len(raw_collocations[word])):
                (w1_raw, w2_raw), t_raw = raw_collocations[word][i]
                (w1_lemma, w2_lemma), t_lemma = lemmatized_collocations[word][i]

                rows.append({
                    'Word': word,
                    'Raw Collocation': f'{w1_raw} {w2_raw}',
                    'Raw T-score': t_raw,
                    'Lemmatized Collocation': f'{w1_lemma} {w2_lemma}',
                    'Lemmatized T-score': t_lemma
                })

  
        df = pd.DataFrame(rows)


        df.to_excel(file_name, index=False)