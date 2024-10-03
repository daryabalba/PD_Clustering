# PD_Clustering

# Brief description

The repository contains the code of HSE Center of Language and Brain's project entitled Digital Instruments for Treating and Evaluating Language and Mental Disorders ([Цифровые инструменты для коррекции и оценки речевых и психических расстройств](https://stratpro.hse.ru/resilient-brain/#subproject2)). It contains the preprocessing of language materials, the implementation of clustering algorithm according to [Lundin et. al](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X) and metrics calculation.

# Краткое описание проекта

Данный репозиторий создан в рамках проекта "[Цифровые инструменты для коррекции и оценки речевых и психических расстройств](https://stratpro.hse.ru/resilient-brain/#subproject2)" Центра Языка и Мозга НИУ ВШЭ - Нижний Новгород. 

**Репозиторий содержит следующие блоки кода**: 

- предобработка языковых материалов,
- реализация алгоритма кластеризации согласно работе [Lundin et. al](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X)
- подсчет метрик кластеризации.

Более подробное описание каждого блока приведено ниже.

# Описание структуры

1. Описание продукта
2. Предобработка языковых материалов
4. Алгоритм кластеризации
5. Метрики кластеризации

## Предобработка языковых материалов

Для транскрибирования аудиозаписей речи испытуемых использовалась модель машинного обучения [Whisper](https://github.com/openai/whisper). 

После транскрибирования и ручной проверки тексты **предобрабатывались в 4 форматах** :

- токенизация с сохранением всех токенов
- токенизация с удалением стоп-слов
- лемматизация с сохранением всех токенов
- лемматизация с удалением стоп-слов

Для предобработки создан класс Preprocessing (см. src). Класс содержит функцию для **токенизации** приходящего на вход текста **в двух вариантах**:

```python
  def tokenize(self, text):
    """
    Getting all tokens, except punctuation marks
    Return: list of tokens with stopwords, list of tokens without stopwords
    """
    tokens = word_tokenize(text)
    tokens_w_stops = [i.lower() for i in tokens if (i not in punctuation)]
    tokens_wo_stops = [i.lower() for i in tokens if (i not in punctuation) and (i not in self.stop_words)]
    return tokens_w_stops, tokens_wo_stops
```

А также для **лемматизации** приходящего на вход текста в двух вариантах:

```python
  def lemmatize(self, text):
    """
    Getting lemmas from text with and without stopwords
    Return: list of lemmas with stopwords, list of lemmas without stopwords
    """
    doc = self.nlp(text)
    lemmas = [token.lemma_.lower() for token in doc if (token.text not in punctuation)]
    lemmas_without_stops = [token.lemma_.lower() for token in doc if (token.text not in punctuation) and (token.text not in self.stop_words)]
    return lemmas, lemmas_without_stops
```

Предобработанные токены не содержат знаков препинания, приведены к нижнему регистру, список стоп-слов взят из [библиотеки NLTK](https://www.nltk.org/).

## Алгоритм кластеризации
