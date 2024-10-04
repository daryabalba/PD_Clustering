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
4. [Метрики кластеризации](#метрики-кластеризации)

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

1. [DataExtraction module](#dataextraction-module)
2. [Models module](#models-module)
3. [Vectorizer module](#vectorizer-module)
4. [Clusterizer module](#clusterizer-module)
5. [ClustersData module](#clustersdata-module)

### DataExtraction module

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

### Models module

### Vectorizer module

### Clusterizer module

### ClustersDataSaver module

Сохраняет в DataFrame **self.healthy_data** и **self.impediment_data** кластеры по каждому формату хранения речевого материала (*токены и леммы со стоп-словами, токены и лемы без стоп-слов;* подробнее см. [Предобработка языковых материалов](#предобработка-языковых-материалов)) и метрики кластеризации (подробнее см. [Метрики кластеризации](#метрики-кластеризации)).

Получает на вход модель извлечения данных (подробне см. [DataExtraction module](#dataextraction-module) и модель векторизации (gensim.models.fasttext.FastTextKeyedVectors, подробнее см. [Models module](#models-module)).

**Содержит следующие функции:**

1. Функция подсчета **количества переключений между кластерами**

```python
def count_num_switches(self,
                       sheet_name: str,
                       category: str) -> None:
        """
        Count number of switches for each cell
        """
        if sheet_name == 'healthy':
            new_column_name = f'Switch_number_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(lambda x: len(x) - 1)

        else:
            new_column_name = f'Switch_number_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(lambda x: len(x) - 1)
```

2. Функция подсчета **среднего размера кластеров**

```python
def count_mean_cluster_size(self,
                            sheet_name: str,
                            category: str) -> None:
        """
        Count mean cluster size for each row
        """
        if sheet_name == 'healthy':
            new_column_name = f'Mean_cluster_size_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(self.avg_cluster_size)

        else:
            new_column_name = f'Mean_cluster_size_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(self.avg_cluster_size)
```

3. Функция подсчета **среднего расстояния между кластерами**

```python
def count_mean_distances(self,
                         sheet_name: str,
                         category: str):
    """
    Counting distances for all columns
    """
    if sheet_name == 'healthy':
        new_column_name = f'Mean_distance_{category}'
        self.healthy_data[new_column_name] = self.healthy_data[category].apply(self.avg_cluster_distance)

    else:
new_column_name = f'Mean_distance_{category}'
        self.impediment_data[new_column_name] = self.impediment_data[category].apply(self.avg_cluster_distance)
```

4. Функция подсчета [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

```python
def silhouette_score(self, cluster_sequence):
  silhouette_coefs = []

  for idx, cluster in enumerate(cluster_sequence):
      for word_1 in cluster:

          a = sum(self.model.similarity(word_1, word_2)
                  for word_2 in cluster if word_1 != word_2) / len(cluster)

          if idx != len(cluster_sequence) - 1:
              b = sum(self.model.similarity(word_1, word_2)
                      for word_2 in cluster_sequence[idx + 1]) / len(cluster_sequence[idx + 1])
          else:
              b = sum(self.model.similarity(word_1, word_2)
                      for word_2 in cluster_sequence[idx - 1]) / len(cluster_sequence[idx - 1])

          s = (b - a) / max(a, b)
          silhouette_coefs.append(s)

  if silhouette_coefs:
      return sum(silhouette_coefs) / len(silhouette_coefs)
  return np.NaN
```

5. Функция подсчета t-score внутри одного кластера

```python
@staticmethod
def cluster_t_score(f_n, f_c, f_nc, N):
    if f_nc == 0:
        return 0
    numerator = f_nc - f_n * f_c / N
    denominator = np.sqrt(f_nc)
    return numerator / denominator

def avg_cluster_t_score(self, cell, column_clusters):
    all_words = ' '.join([word for cell in column_clusters for cluster in cell for word in cluster])
    N = len(all_words)

    cell_t_scores = []
    for cluster in cell:
        all_wordpairs = list(permutations(cluster, 2))

        pairwise_t_scores = []
        for wordpair in all_wordpairs:
            f_n = all_words.count(wordpair[0])
            f_c = all_words.count(wordpair[1])
            f_nc = all_words.count(' '.join((wordpair[0], wordpair[1])))
            f_nc += all_words.count(' '.join((wordpair[1], wordpair[0])))

            t_score = self.cluster_t_score(f_n, f_c, f_nc, N)
            pairwise_t_scores.append(t_score)

        cell_t_scores.extend(pairwise_t_scores)

return sum(cell_t_scores)
```

6. Функция сохранения в Excel файл по заданному пути path

```python
def save_excel(self, path) -> None:
    """
    Saving data with clusters to an Excel file
    """
    with pd.ExcelWriter(path) as writer:
        self.healthy_data.to_excel(writer, sheet_name='healthy', index=False)
        self.impediment_data.to_excel(writer, sheet_name=self.impediment_type, index=False)
```

Более подробное описание каждой метрики см. в блоке [Метрики кластеризации](#метрики-кластеризации).

## Метрики кластеризации

Для оценки качества кластеризации использовались следующие метрики:

- средний **размер кластера** (среднее количество слов, входящих в кластер);
- среднее **расстояние между кластерами** (считается как расстояние между центроидами кластеров; за центроид берется среднее значение векторов, находящихся в одном кластере);
- 
- Среднее значение **t-score в кластере** для каждого человека (метрика, отображающая насколько неслучайной является сила ассоциации между коллокатами; в качестве коллокатов берутся все последовательности из двух слов внутри одного кластера).
- Среднее значение **silhouette-score** (метрика, отображающая, насколько близок объект к своему кластеру по сравнению с другими кластерами: чем больше значение данной метрики, тем ближе объект к своему собственному кластеру).
