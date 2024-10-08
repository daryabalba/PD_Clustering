import pandas as pd
from collocation_finder import CollocationFinder
from text_processor import TextProcessor

def main():
    # The path to the table with the texts
    file_path = 'schizotypal.disorder.xlsx'
    column_name = 'transcript'

    # A list of words to search for collocations
    words = ['это', 'очень', 'наверное', 'подарок', 'такой', 'помнить', 'самый', 'просто', 'мой', 'который',
             'косметика', 'родитель', 'лежать', 'оно', 'пойти', 'оказаться', 'любимый', 'фотоаппарат',
             'выбраться', 'подарить']

    # Uploading texts
    def load_texts_from_excel(file_path, column_name):
        df = pd.read_excel(file_path)
        return df[column_name].dropna().tolist()
    
    texts = load_texts_from_excel(file_path, column_name)

    # Text processing and collocation acquisition
    collocation_finder = CollocationFinder()
    text_processor = TextProcessor()
    raw_collocations, lemmatized_collocations = collocation_finder.process_texts(texts, words, text_processor)

    # Saving the results
    collocation_finder.save_to_excel(words, raw_collocations, lemmatized_collocations)

if name == "__main__":
    main()