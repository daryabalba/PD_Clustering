# PD_Clustering

# Brief description

The repository contains the code of HSE Center of Language and Brain's project entitled Speech profile tool for generalized linguistic personality in the diagnosis of mental disorders ([Инструмент для создания речевого профиля обобщенной языковой личности для диагностики ментальных расстройств](https://stratpro.hse.ru/resilient-brain/#subproject14)). It contains the preprocessing of language materials, the implementation of clustering algorithm according to [Lundin et. al](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X) and metrics calculation.

# Краткое описание проекта

Данный репозиторий создан в рамках проекта "[Инструмент для создания речевого профиля обобщенной языковой личности для диагностики ментальных расстройств](https://stratpro.hse.ru/resilient-brain/#subproject14)" Центра Языка и Мозга НИУ ВШЭ - Нижний Новгород. 

**Репозиторий содержит следующие блоки кода**: 

- предобработка языковых материалов,
- реализация алгоритма кластеризации согласно работе [Lundin et. al](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X)
- подсчет метрик кластеризации.

Более подробное описание каждого блока приведено ниже.

# Описание структуры

1. Описание продукта
2. [Предобработка языковых материалов](#предобработка-языковых-материалов)
3. [Алгоритм кластеризации](#алгоритм-кластеризации)
4. Метрики кластеризации

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
		tokens_w_stops = ', '.join([i.lower() for i in tokens if (i not in punctuation)])
		tokens_wo_stops = ', '.join([i.lower() for i in tokens if (i not in punctuation) and (i not in self.stop_words)])
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
    lemmas = ', '.join([token.lemma_.lower() for token in doc if (token.text not in punctuation)])
    lemmas_without_stops = ', '.join([token.lemma_.lower() for token in doc if (token.text not in punctuation) and (token.text not in self.stop_words)])
    return lemmas, lemmas_without_stops
```

Предобработанные токены не содержат знаков препинания, приведены к нижнему регистру, список стоп-слов взят из [библиотеки NLTK](https://www.nltk.org/).

## Алгоритм кластеризации

1. [DataExtraction module](#1.-dataextraction_module)
2. [ClustersData module](clustersdata_module)
3. [Clusterizer module](clusterizer_module)
4. [Vectorizer module](vectorizer_module)

### 1. DataExtraction module

Извлекает информацию из таблицы Excel, содержащей предобработанные тексты:

- список ID респондентов

```python
def get_ids(self, sheet_name: str = 'healthy') -> pd.DataFrame:
	  """
	  Getting ID column
	  """
	  if sheet_name == 'healthy':
	      return self.dataset_norm['speakerID']
	  return self.dataset_pd['ID']
```

- список предобработанных текстов в 4 вариантах (в формате Series)

```python
def get_series(self,
               sheet_name: str,
               category: str) -> pd.DataFrame:
    """
    Getting one of 8 columns:
      from one of the 2 pages of the dataset
      from one of the 4 categories

    sheet_name: healthy | PD
    category: tokens | tokens_without_stops | lemmas | lemmas_without_stops
    """
    if sheet_name == 'healthy':
        return self.dataset_norm[category]

    return self.dataset_pd[category]
```

### 2. ClustersData module

### 3. Clusterizer module

### 4. Vectorizer module
