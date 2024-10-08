# PD_Clustering

# Brief description

The repository contains the code of HSE Center of Language and Brain's project entitled Speech profile tool for generalized linguistic personality in the diagnosis of mental disorders ([Инструмент для создания речевого профиля обобщенной языковой личности для диагностики ментальных расстройств](https://stratpro.hse.ru/resilient-brain/#subproject14)). It contains the preprocessing of language materials, the implementation of clustering algorithm according to [Lundin et. al](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X) and metrics calculation. In addition, it includes several text analysis modules that allow us to calculate various stylostatictic metrics, as well as search for keywords, collocations, bigrams and trigrams.

# Краткое описание проекта

Данный репозиторий создан в рамках проекта "[Инструмент для создания речевого профиля обобщенной языковой личности для диагностики ментальных расстройств](https://stratpro.hse.ru/resilient-brain/#subproject14)" Центра Языка и Мозга НИУ ВШЭ - Нижний Новгород. 

**Репозиторий содержит следующие блоки кода**: 

- предобработка языковых материалов,
- реализация алгоритма кластеризации согласно работе [Lundin et. al](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X)
- подсчет метрик кластеризации,
- подсчет стилостатистики,
- поиск ключевых слов,
- поиск коллокаций с ключевыми словами,
- поиск n-грамм (биграмм и триграмм)

Более подробное описание каждого блока приведено ниже.

# Описание структуры

1. [Описание продукта](#описание-продукта)
2. [Предобработка языковых материалов](#предобработка-языковых-материалов)
3. [Алгоритм кластеризации](#алгоритм-кластеризации)
4. [Метрики кластеризации](#метрики-кластеризации)
5. [Подсчет стилостатистики](#подсчет-стилостатистики)
6. [Поиск ключевых слов](#поиск-ключевых-слов)
7. [Поиск коллокаций с ключевыми словами](#поиск-коллокаций-с-ключевыми-словами)
8. [Поиск n-грамм (биграмм и триграмм)](#поиск-n-грамм)


## Описание продукта
Здесь будет схема исследования, но пока ее нет. Подробное описание проделанной работы расположено по [ссылке]().

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

Предобработанные токены не содержат знаков препинания, приведены к нижнему регистру, список стоп-слов взят из [библиотеки NLTK](https://www.nltk.org/). Для лемматизации использовалась библиотека [SpaCy](https://spacy.io/) (модель [ru_core_news_sm](https://spacy.io/models/ru#ru_core_news_sm)). В результате предобработки получен [датасет](https://github.com/daryabalba/PD_Clustering/blob/main/data/control_pd_preprocessed.xlsx), хранящийся в папке data.

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

Используется для импорта модели Geowac (***geowac_lemmas_none_fasttextskipgram_300_5_2020*)** из корпуса [RusVectores](https://rusvectores.org/ru/models/). Модель построена на корпусе Geowac с использованием fasttext.

### Vectorizer module

Получает на вход модель для векторизации. Записывает векторы меток **Beginning of Sequence** (BOS), **Pre-End of Sequence** (PEOS) и **End of Sequence** (EOS) в словарь векторов. Данные метки необходимы для последующей кластеризации, т.к. первое слово имеет косинусное расстояние равное 0 за неимением предшествующих (косинусное расстояние считается между текущим словом и предыдущим).

**Структура словаря**: *ключ — слово, значение — вектор, полученный из модели векторизации*. Словарь строится по корпусу текстов и обновляется при появлении новых слов.

**Содержит следующие функции**:

1. Функция обновления внутреннего словаря класса `update_dict(self, words: str)`
2. Функция создания последовательности токенов с метками начала и конца последовательности `get_sequence(words_string: str)`
3. Функция getter словаря `get_dictionary(self)`

### Clusterizer module

**TLDR**: Модуль содержит функции кластеризации текста и оценки качества кластеризации (*Davies Bouldin index + Silhouette Score*)

Кластеризация проводится по формуле, предложенной в работе [Lundin et al.](https://www.sciencedirect.com/science/article/abs/pii/S016517812200018X), согласно которой для последовательности слов $A, B, C, D$ **переключение находится после слова** $B$ в случае, **когда** $S(A, B) > S(B, C)$ и $S(B, C) < S(C, D)$, где $S(A, B)$ – косинусная близость между векторами слов $A$ и $B$.

Косинусное расстояние рассчитывается по формуле: `np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))`, где v1, v2 — векторы слов. В случае с моделью Geowac используется метод similarity.

**Содержит следующие функции**:

1. Функция подсчета косинусного расстояния `get_cosine_similarity(self, w1, w2)`
2. Функция кластеризации `cluster(self, word_sequence: list[str])` 
3. Функция подсчета расстояния между центроидами `_custom_similarity(embedding_1, embedding_2)`. Центроид — вектор среднего значения последовательностей embedding_1, embedding_2
4. Функция расчета [Davies Bouldin index](https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index) `davies_bouldin_index(self, cluster_sequence: list[list[str]])`
5. Функция подсчета [silhouette score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient) `silhouette_score(self, cluster_sequence: list[list[str]])`
6. Функция оценки результатов кластеризации по всем метрикам `evaluate_clustering(DB_values_page: list[float], silhouette_values: list[float])`

### ClustersDataSaver module

Сохраняет в DataFrame **self.healthy_data** и **self.impediment_data** кластеры по каждому формату хранения речевого материала (*токены и леммы со стоп-словами, токены и лемы без стоп-слов;* подробнее см. [Предобработка языковых материалов](#предобработка-языковых-материалов)) и метрики кластеризации (подробнее см. [Метрики кластеризации](#метрики-кластеризации)).

Получает на вход модель извлечения данных (подробне см. [DataExtraction module](#dataextraction-module) и модель векторизации (gensim.models.fasttext.FastTextKeyedVectors, подробнее см. [Models module](#models-module)).

**Содержит следующие функции:**

1. Функция подсчета **количества переключений между кластерами** `count_num_switches(self, sheet_name: str, category: str)`
2. Функция подсчета **среднего размера кластеров** `avg_cluster_size(row: pd.Series)`
3. Функция подсчета **среднего расстояния между кластерами** `avg_cluster_distance(self, cluster_sequence)`
4. Функция подсчета [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) `silhouette_score(self, cluster_sequence)`
5. Функция подсчета [t-score](https://blogs.helsinki.fi/slavica-helsingiensia/files/2019/11/sh34-21.pdf) внутри одного `кластера cluster_t_score(f_n, f_c, f_nc, N)`
6. Функция подсчета среднего t-score `avg_cluster_t_score(self, cell, column_clusters)`
7. Функция **сохранения в Excel файл** по заданному пути path `save_excel(self, path)`

Более подробное описание каждой метрики см. в блоке [Метрики кластеризации](#метрики-кластеризации).

## Метрики кластеризации

Для последующего анализа результатов кластеризации использовались следующие метрики:

- среднее **количество переключений** между кластерами;
- средний **размер кластера** (среднее количество слов, входящих в кластер);
- среднее **расстояние между кластерами** (считается как расстояние между центроидами кластеров; за центроид берется среднее значение векторов, находящихся в одном кластере);
- среднее значение **t-score в кластере** для каждого человека (метрика, отображающая насколько неслучайной является сила ассоциации между коллокатами; в качестве коллокатов берутся все последовательности из двух слов внутри одного кластера).
- среднее значение **silhouette-score** (метрика, отображающая, насколько близок объект к своему кластеру по сравнению с другими кластерами: чем больше значение данной метрики, тем ближе объект к своему собственному кластеру).

## Результаты кластеризации

Результаты кластеризации хранятся в папке [result](https://github.com/daryabalba/PD_Clustering/tree/main/result).

## Подсчет стилостатистики

После транскрибирования и ручной проверки тексты не нуждаются в дальнейшей предобработке для подсчета стилостатистики. 

### SentenceMetrics module

Рассчитывает статистику по предложениям.

**Содержит следующие функции:**

1. Функция подсчета средней длины предложения в словах `avg_sentence_len(self)`

### WordMetrics module

Рассчитывает статистику по словам.

**Содержит следующие функции:**

1. Функция подсчета средней длины слова в символах `avg_word_len(self)`

### ReadabilityMetrics module

Рассчитывает идексы удобочитаемости и структурную сложность текста.

**Содержит следующие функции:**

1. Функция подсчета идекса удобочитаемости Флеша-Кинкейда ([Flesch–Kincaid readability tests](https://ling.hse.ru/news/490474121.html)) `flesch_kincaid_grade(self)`
3. Функция подсчета идекса туманности Ганнинга ([Gunning fog index](https://ling.hse.ru/news/490474121.html)) `gunning_fog_index(self)`
4. Функция подсчета идекса лексического разнообразия ([Type Token Ratio](https://elib.utmn.ru/jspui/bitstream/ru-tsu/7382/1/humanitates_2020_1_20_34.pdf)) `ttr(self)`

### StylisticMetrics module

Рассчитывает [показатели](https://uploads.philology.by/golovin-b.n_.-jazyk-i-statistika-1971_.pdf), которые используются в лингвистическом анализе текста для количественной оценки его содержания, структуры и стиля. Эти коэффициенты применяются для характеристики текста с точки зрения его смысловой наполненности, динамики, связности и других параметров, что помогает глубже понять его структуру и качество.

**Содержит следующие функции:**

1. Функция подсчета коэффициента предметности (Pr) `subjectivity_coefficient(self)` – отношение суммы существительных и местоимений к сумме прилагательных и глаголов. Этот коэффициент оценивает, насколько текст насыщен конкретными понятиями, терминами и предметными номинациями.
2. Функция подсчета коэффициента качественности (Qu) `quality_coefficient(self)` – отношение суммы прилагательных и наречий к сумме глаголов и существительных. Этот коэффициент оценивает наличие в тексте качественных характеристик, выраженных через прилагательные.
3. Функция подсчета коэффициента активности (Ac) `activity_coefficient(self)` – отношение суммы глаголов и глагольных форм к количеству слов в тексте. Этот коэффициент измеряет динамику текста, оценивая, насколько часто используются глаголы и другие части речи, выражающие действие.
4. Функция подсчета коэффициента динамизма (Din) `dynamism_coefficient(self)` –  отношение суммы глаголов и глагольных форм к сумме существительных, прилагательных и местоимений. Динамизм текста характеризует темп и степень изменения событий или состояний в тексте.
5. Функция подсчета коэффициента связности текста (Con) `cohesion_coefficient(self)` – отношение суммы предлогов и союзов к числу предложений. Этот коэффициент оценивает, насколько текст логически структурирован и как хорошо связаны между собой отдельные части текста, предложения и абзацы.

## Поиск ключевых слов

### TextProcessor module

Перед проведением поиска ключевых слов транскрибированные тексты подверглись предварительной обработке. 
`text_processing.py`: Этот файл включает класс `TextProcessor`, который выполняет предварительную обработку текста: очищение, токенизация, удаление стоп-слов и лемматизация.

**Содержит следующие функции:**

1. Функция очистки текста от знаков препинания, чисел и пробелов `clean_text(self)`
2. Функция токенизации `tokenize_text(self, text)`
3. Функция удаления стоп-слов из списка токенов `remove_stopwords(self, tokens)`
4. Функция лемматизации `lemmatize_text(self, text)`
5. Функция полного цикла обработки текста `process_text(self)`

Эти функции позволяют подготовить текст для дальнейшего анализа, убрав лишние элементы и нормализовав слова. Предобработанные токены не содержат знаков препинания, приведены к нижнему регистру, список стоп-слов взят из [библиотеки NLTK](https://www.nltk.org/). Для лемматизации использовалась библиотека [pymorphy2](https://pypi.org/project/pymorphy2/).  

### MetricsCalculator module

Рассчитывает различные метрики длля анализа ключевых слов.

**Содержит следующие функции:**
1. Функция рассчета статистики Log-Likelihood для ключевых слов `calculate_log_likelihood(f_c, f_ref, e_c, e_r)`
2. Функция рассчета t-score для частоты слов`calculate_t_score(f_c, e_c))`
3. Функция рассчета MI для слов `calculate_mutual_information(f_c, f_ref, N_c, N_total)`
4. Функция рассчета коэффициента Dice для оценки схожести частоты использования слов в корпусах `calculate_dice_coefficient(f_c, f_ref)`
5. Функция поиска ключевых слов в корпусе текстов и рассчета для них метрик (Log-Likelihood, t-score, MI, Dice, Frequency) сравнивая частоты с референсным корпусом `find_keywords_metrics(corpus_texts, reference_texts, text_processor)`. В качестве референсного корпуса использовался корпус [OpenCorpora](https://www.opencorpora.org/).

## Поиск коллокаций с ключевыми словами

### CollocationFinder module

Производит поиск коллокаций с заданными ключевыми словами по метрике [t-score](https://blogs.helsinki.fi/slavica-helsingiensia/files/2019/11/sh34-21.pdf).

**Содержит следующие функции:**
1. Функция поиска биграмм для заданного слова `get_collocations(self, text, word, top_n=20)`
2. Функция поиска коллокаций для списка слов в нелемматизированном и лемматизированном тексте `process_texts(self, texts, words, top_n=20, text_processor)`
3. Функция сохранения найденных коллокаций в Exel-файл `save_to_excel(self, words, raw_collocations, lemmatized_collocations, file_name='collocations.xlsx')`

## Поиск n-грамм (биграмм и триграмм)

### NgramExtractor module

Извлекает n-граммы (биграммы и триграммы) из текстов (как из исходных, так и лемматизированных).

**Содержит следующие функции:**
1. Функция создает n-граммы с помошью `CountVectorizer` и воззвращает таблицу, где каждая строка содержит n-грамму и ее частоту `create_ngrams_df(self, text, n=2)`
2. Функция вызывает этот процесс для биграмм и триграмм как для исходного текста, так и для лемматизированного, возращая четыре таблицы с результатами `extract_ngrams(self, raw_text, lemmatized_text)`
3. Функция сорнаняет каждую из четырех таблиц в отдельные листы Exel `save_to_excel(self, bigrams_df, trigrams_df, lemmatized_bigrams_df, lemmatized_trigrams_df, file_name='ngrams.xlsx')`

## Результаты стилостатистики

Результаты стилостатистики хранятся в папке [stylostatistics](https://github.com/daryabalba/PD_Clustering/tree/main/result/stylostatistics).
