class WordMetrics:
    def __init__(self, text) -> None:
      self.text = text

    def avg_word_len(self):
        """
          Calculate the average word length in letters
        """
        words = self.text.split()
        if len(words) == 0:
          return 0
        return sum(len(word) for word in words) / len(words)