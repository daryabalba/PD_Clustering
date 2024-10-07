import re
from ruts import ReadabilityStats

class ReadabilityMetrics:
    def __init__(self, text) -> None:
        self.text = text

    def flesch_kincaid_grade(self):
        """
        Flesch-Kincaid readability grade level
        """
        rs = ReadabilityStats(self.text)
        return rs.get_stats()['flesch_kincaid_grade']

    def gunning_fog_index(self):
        """
        Gunning fog index for readability
        """
        sentences = re.split(r'[.!?]', self.text)
        words = re.findall(r'\b\w+\b', self.text)
        total_words = len(words)
        total_sentences = len(sentences)
        complex_words = self.count_complex_words(words)
        if total_sentences == 0:
            return 0
        return 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))

    def count_complex_words(self, words):
        """
        Count the number of complex words (with 3+ syllables)
        """
        return len([word for word in words if self.count_syllables(word) >= 3])

    def count_syllables(self, word):
        """
        Count the number of syllables in a word
        """
        vowels = "aeiouy"
        syllables = 0
        prev_char = ''
        for char in word.lower():
          if char in vowels and prev_char not in vowels:
            syllables += 1
          prev_char = char
        if word.endswith("e"):
          syllables -= 1
        return max(syllables, 1)

    def ttr(self):
        """
        Type-token ratio (TTR) for lexical diversity
        """
        words = re.findall(r'\b\w+\b', self.text.lower())
        total_words = len(words)
        unique_words = len(set(words))
        return unique_words / total_words if total_words > 0 else 0