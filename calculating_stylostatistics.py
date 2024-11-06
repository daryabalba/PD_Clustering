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
    sentence_metrics = [SentenceMetrics(text) for text in df['text']]
    word_metrics = [WordMetrics(text) for text in df['text']]
    readability_metrics = [ReadabilityMetrics(text) for text in df['text']]
    stylistic_metrics = [StylisticMetrics(text) for text in df['text']]

    # Sentence metrics
    df['avg_sent_len'] = [metrics.avg_sentence_len() for metrics in sentence_metrics]

    # Word metrics
    df['avg_word_len'] = [metrics.avg_word_len() for metrics in word_metrics]

    # Readability metrics
    df['FK_readability'] = [metrics.flesch_kincaid_grade() for metrics in readability_metrics]
    df['gunning_fog_index'] = [metrics.gunning_fog_index() for metrics in readability_metrics]
    df['ttr'] = [metrics.ttr() for metrics in readability_metrics]

    # Stylistic metrics
    df['Pr'] = [metrics.subjectivity_coefficient() for metrics in stylistic_metrics]
    df['Qu'] = [metrics.quality_coefficient() for metrics in stylistic_metrics]
    df['Ac'] = [metrics.activity_coefficient() for metrics in stylistic_metrics]
    df['Din'] = [metrics.dynamism_coefficient() for metrics in stylistic_metrics]
    df['Con'] = [metrics.cohesion_coefficient() for metrics in stylistic_metrics]

    # Save result
    df.to_excel('results.xlsx', index=False)

if __name__ == '__main__':
    main()
