import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class NgramExtractor:
    def create_ngrams_df(self, text, n=2):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        X = vectorizer.fit_transform([text])
        ngrams = vectorizer.get_feature_names_out()
        frequencies = X.toarray().sum(axis=0)

        return pd.DataFrame({'Ngram': ngrams, 'Frequency': frequencies})

    def extract_ngrams(self, raw_text, lemmatized_text):
        bigrams_df = self.create_ngrams_df(raw_text, n=2)
        trigrams_df = self.create_ngrams_df(raw_text, n=3)
        lemmatized_bigrams_df = self.create_ngrams_df(lemmatized_text, n=2)
        lemmatized_trigrams_df = self.create_ngrams_df(lemmatized_text, n=3)

        return bigrams_df, trigrams_df, lemmatized_bigrams_df, lemmatized_trigrams_df

    def save_to_excel(self, bigrams_df, trigrams_df, lemmatized_bigrams_df, lemmatized_trigrams_df, file_name='ngrams.xlsx'):
        with pd.ExcelWriter(file_name) as writer:
            bigrams_df.to_excel(writer, sheet_name='Bigrams', index=False)
            trigrams_df.to_excel(writer, sheet_name='Trigrams', index=False)
            lemmatized_bigrams_df.to_excel(writer, sheet_name='Lemmatized_Bigrams', index=False)
            lemmatized_trigrams_df.to_excel(writer, sheet_name='Lemmatized_Trigrams', index=False)
        print(f'Ngrams saved to {file_name}')