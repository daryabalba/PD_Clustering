import pandas as pd
from src.xml_parser import XMLParser
from src.text_processing import TextProcessor
from src.metrics_calculator import MetricsCalculator

def main():
    # Preparation of the reference corpus
    xml_parser = XMLParser('annot.opcorpora.no_ambig.xml')
    X, Y = xml_parser.extract_sentences()
    reference_text = xml_parser.convert_to_text(X)

    # Uploading data
    df = pd.read_excel('data.xlsx')
    corpus_texts = df['transcript'].tolist()

    # Text preprocessing
    text_processor = TextProcessor()
    reference_text = text_processor.process_text(reference_text)

    # Calculating metrics
    metrics_calculator = MetricsCalculator()
    keywords_table = metrics_calculator.find_keywords_metrics(corpus_texts, [reference_text], text_processor)

    # Output of results
    keywords_table.to_excel('keywords_metrics.xlsx')

if name == "__main__":
    main()
