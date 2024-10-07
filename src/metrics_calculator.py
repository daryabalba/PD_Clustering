import math
from collections import Counter
import pandas as pd

class MetricsCalculator:
    @staticmethod
    def calculate_log_likelihood(f_c, f_ref, e_c, e_r):
        """Log-Likelihood."""
        return 2 * (f_c * math.log(f_c / e_c) + f_ref * math.log(f_ref / e_r))

    @staticmethod
    def calculate_t_score(f_c, e_c):
        """T-Score."""
        return (f_c - e_c) / math.sqrt(f_c)

    @staticmethod
    def calculate_mutual_information(f_c, f_ref, N_c, N_total):
        """Mutual Information (MI)."""
        p_wc = f_c / N_c
        p_w = (f_ref + f_c) / N_total
        return math.log2(p_wc / p_w) if p_w != 0 else 0

    @staticmethod
    def calculate_dice_coefficient(f_c, f_ref):
        """Dice."""
        return (2 * f_c) / (f_c + f_ref) if (f_c + f_ref) != 0 else 0

    @staticmethod
    def find_keywords_metrics(corpus_texts, reference_texts, text_processor):
        """Поиск ключевых слов и расчет различных метрик для них."""
        corpus_words = []
        for text in corpus_texts:
            corpus_words.extend(text_processor.preprocess_and_lemmatize(text))

        reference_words = []
        for text in reference_texts:
            reference_words.extend(text_processor.preprocess_and_lemmatize(text))

        corpus_freq = Counter(corpus_words)
        reference_freq = Counter(reference_words)

        N_corpus = sum(corpus_freq.values())
        N_reference = sum(reference_freq.values())
        N_total = N_corpus + N_reference

        keywords = []
        for word in corpus_freq:
            f_corpus = corpus_freq[word]
            f_reference = reference_freq.get(word, 0)

            
            e_c = (f_corpus + f_reference) * (N_corpus / (N_corpus + N_reference))
            e_r = (f_corpus + f_reference) * (N_reference / (N_corpus + N_reference))

            lr = MetricsCalculator.calculate_log_likelihood(f_corpus, f_reference, e_c, e_r)
            t_score = MetricsCalculator.calculate_t_score(f_corpus, e_c)
            mi = MetricsCalculator.calculate_mutual_information(f_corpus, f_reference, N_corpus, N_total)
            dice = MetricsCalculator.calculate_dice_coefficient(f_corpus, f_reference)

            ratio = (f_corpus + 1) / (f_reference + 1)
            keywords.append((word, f_corpus, f_reference, ratio, lr, t_score, mi, dice))

        keywords_df = pd.DataFrame(keywords, columns=['Word', 'Freq Corpus', 'Freq Reference', 'Frequency Ratio (freq_corpus_control / freq_opencorpora)', 'Log-Likelihood', 'T-Score', 'MI', 'Dice'])
        keywords_df = keywords_df.sort_values(by='Log-Likelihood', ascending=False).reset_index(drop=True)

        return keywords_df