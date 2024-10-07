class SentenceMetrics:
    def __init__(self, text) -> None:
        self.text = text

    def avg_sentence_len(self):
        """
          Calculate the average sentence length in words
        """
        sentences = self.text.split(".")
        words = self.text.split(" ")
        if sentences[-1]=="":
          average_sentence_length = len(words) / len(sentences)-1
        else:
          average_sentence_length = len(words) / len(sentences)
        return average_sentence_length