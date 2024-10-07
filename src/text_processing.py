import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import pymorphy2

nltk.download('punkt')
nltk.download('stopwords')

class TextProcessor:
    def __init__(self, text) -> None:
        
        self.text = text.lower()
        self.stop_words = set(stopwords.words("russian"))
        self.morph = pymorphy2.MorphAnalyzer()

    def clean_text(self):
        text = self.text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        return text

    def tokenize_text(self, text):
        """
        Tokenization and punctuation filtering
        """
        tokens = word_tokenize(text, language="russian")
        tokens = [w.strip(punctuation) for w in tokens if w.isalpha()]
        return tokens

    def remove_stopwords(self, tokens):
        """
        Deleting stop words
        """
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize_text(self, text):
        """
        Lemmatization of the text
        """
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        lemmatized_words = [self.morph.parse(word)[0].normal_form for word in tokens]
        return ' '.join(lemmatized_words)

    def process_text(self):
        """
        Full text processing: clearing, tokenization, removal of stop words and lemmatization
        """
        cleaned_text = self.clean_text()
        return self.lemmatize_text(cleaned_text)