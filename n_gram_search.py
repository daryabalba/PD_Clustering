from src.text_processing import TextProcessor
from src.ngram_extractor import NgramExtractor

def main():
    # The path to the table with the texts
    file_path = 'schizotypal.disorder.xlsx'
    text_column = 'transcript'

    # Initializing the text handler
    def load_texts(file_path, text_column):
        df = pd.read_excel(self.file_path)
        return df[self.text_column].dropna().tolist()

    def get_combined_text(file_path, text_column):
        texts = self.load_texts()
        return " ".join(texts)

    text_processor = TextProcessor(file_path, text_column)

    
    combined_text = get_combined_text(file_path, text_column)
    lemmatized_text = text_processor.lemmatize_text(combined_text)

    # Extracting bigrams and trigrams
    ngram_extractor = NgramExtractor()
    bigrams_df, trigrams_df, lemmatized_bigrams_df, lemmatized_trigrams_df = ngram_extractor.extract_ngrams(combined_text, lemmatized_text)

    # Saving the results in Excel
    ngram_extractor.save_to_excel(bigrams_df, trigrams_df, lemmatized_bigrams_df, lemmatized_trigrams_df)

if name == "__main__":
    main()
