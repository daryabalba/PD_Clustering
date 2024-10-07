import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class StylisticMetrics:
    def __init__(self, text) -> None:
       self.text = text
       self.words = word_tokenize(text.lower())
       self.tagged_words = pos_tag(self.words)

    def subjectivity_coefficient(self):
        """
        Calculate subjectivity coefficient based on POS tagging
        """
        subject_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        subject_words = [word for word, tag in self.tagged_words if tag in subject_tags]
        meaningful_words = [word for word in self.words if word.isalpha() and word not in stopwords.words('russian')]
        return len(subject_words) / len(meaningful_words) if len(meaningful_words) > 0 else 0

    def quality_coefficient(self):
        """
        Calculate quality coefficient as the ratio of adjectives/adverbs to nouns/verbs
        """
        adjective_adverb_tags = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        noun_verb_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        adjective_adverb_count = sum(1 for word, tag in self.tagged_words if tag in adjective_adverb_tags)
        noun_verb_count = sum(1 for word, tag in self.tagged_words if tag in noun_verb_tags)
        return adjective_adverb_count / noun_verb_count if noun_verb_count > 0 else float('inf')

    def activity_coefficient(self):
        """
        Calculate activity coefficient based on the ratio of verbs to total words
        """
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        verb_count = sum(1 for word, tag in self.tagged_words if tag in verb_tags)
        return verb_count / len(self.words) if len(self.words) > 0 else 0

    def dynamism_coefficient(self):
        """
        Calculate dynamism coefficient as the ratio of verbs to nouns/adjectives/pronouns
        """
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        noun_adjective_pronoun_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'PRP', 'PRP$', 'WP', 'WP$'}
        verb_count = sum(1 for word, tag in self.tagged_words if tag in verb_tags)
        noun_adjective_pronoun_count = sum(1 for word, tag in self.tagged_words if tag in noun_adjective_pronoun_tags)
        return verb_count / noun_adjective_pronoun_count if noun_adjective_pronoun_count > 0 else float('inf') if verb_count > 0 else 0

    def cohesion_coefficient(self):
        """
        Calculate cohesion coefficient based on the ratio of conjunctions/prepositions to sentences
        """
        sentences = sent_tokenize(self.text)
        preposition_conjunction_tags = {'IN', 'CC'}
        preposition_conjunction_count = sum(1 for word, tag in self.tagged_words if tag in preposition_conjunction_tags)
        return preposition_conjunction_count / len(sentences) if len(sentences) > 0 else 0