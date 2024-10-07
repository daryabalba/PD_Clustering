import pandas as pd
from src.sentence_metrics import SentenceMetrics
from src.word_metrics import WordMetrics
from src.readability_metrics import ReadabilityMetrics
from src.stylistic_metrics import StylisticMetrics
from src.text_processing import TextProcessor

def main():
    # Uploading data
    df = pd.read_excel('data.xlsx')

    # Initializing classes
    text_processing = TextProcessing()
    sentence_metrics = SentenceMetrics()
    word_metrics = WordMetrics()
    readability_metrics = ReadabilityMetrics()
    stylistic_metrics = StylisticMetrics()

    # Text cleaning and preparation
    df['cleaned_text'] = df['transcript'].apply(text_processing.clean_text)

    # Sentence metrics
    df['avg_sent_len'] = df['cleaned_text'].apply(sentence_metrics.avg_sentence_len)

    # Word metrics
    df['avg_word_len'] = df['cleaned_text'].apply(word_metrics.avg_word_len)

    # Readability metrics
    df['FK_readability'] = df['cleaned_text'].apply(readability_metrics.flesch_kincaid_grade)
    df['gunning_fog_index'] = df['cleaned_text'].apply(readability_metrics.gunning_fog_index)
    df['ttr'] = df['cleaned_text'].apply(readability_metrics.ttr)

    # Stylistic metrics
    df['Pr'] = df['cleaned_text'].apply(stylistic_metrics.subjectivity_coefficient)
    df['Qu'] = df['cleaned_text'].apply(stylistic_metrics.quality_coefficient)
    df['Ac'] = df['cleaned_text'].apply(stylistic_metrics.activity_coefficient)
    df['Din'] = df['cleaned_text'].apply(stylistic_metrics.dynamism_coefficient)
    df['Con'] = df['cleaned_text'].apply(stylistic_metrics.cohesion_coefficient)

    # Save result
    df.to_excel('results.xlsx', index=False)

if __name__ == '__main__':
    main()