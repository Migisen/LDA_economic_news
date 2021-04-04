from typing import List

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import (IDF, CountVectorizer, RegexTokenizer,
                                StopWordsRemover)
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType


class LDANewsModel:
    """PySpark LDA model
    """

    def __init__(self, news_df: DataFrame) -> None:
        # TODO: write documentation
        self.tokenizer = RegexTokenizer(inputCol='text', outputCol='words',
                                        pattern=r'[а-яА-Я]+', toLowercase=True,
                                        gaps=False)
        self.count_vectorizer = CountVectorizer(
            inputCol='words', outputCol='raw_features', vocabSize=100_000, minDF=10)
        self.idf = IDF(inputCol='raw_features', outputCol='features')
        self.news: DataFrame = news_df
        self._words_data: DataFrame = self.tokenizer.transform(self.news)
        self._stemmer = SnowballStemmer('russian')
        self._trained_model: LDAModel = None

    def fit(self, num_topics: int = 5, max_iterations: int = 20, thresholds: tuple = (3, 6)) -> LDA:
        # TODO: write documentation
        data, cv_model = self.preprocess(data=self.news, thresholds=thresholds)

        # LDA
        lda_model = LDA(featuresCol='tf', k=num_topics,
                        maxIter=max_iterations)
        result = lda_model.fit(data[['id', 'tf']])
        self._trained_model = result
        return result, cv_model

    def predict(self, data: DataFrame):
        assert self._trained_model is not None, 'Train model first'
        preprocessed_data, cv_model = self.preprocess(data)
        prediction = self._trained_model.transform(preprocessed_data)
        return prediction, cv_model

    def preprocess(self, data: DataFrame, thresholds=(3, 6)) -> DataFrame:
        """Preprocess corpus (tokenization, stopwords, stemming)

        Returns:
            DataFrame: preprocessed spark DataFrame
        """
        # Removing stopwords
        words_data = self.tokenizer.transform(data)
        stop_words = self.calculate_stopwords(
            data=data, words_data=words_data, thresholds=thresholds)
        remover = StopWordsRemover(
            inputCol='words', outputCol='cleared_words', stopWords=stop_words)
        cleared_texts = remover.transform(words_data)

        # Stemming
        stemmingUDF = udf(self.stemUDF, ArrayType(StringType()))
        stemmed_text = cleared_texts.select(col('id'), stemmingUDF(
            col('cleared_words')).alias('stemmed_words'))

        # Final TF-matrix
        count_vectorizer = CountVectorizer(inputCol='stemmed_words', outputCol='tf',
                                           vocabSize=100_000, minDF=10)
        count_model = count_vectorizer.fit(stemmed_text)
        tf_data = count_model.transform(stemmed_text)

        return tf_data, count_model

    def calculate_stopwords(self, data: DataFrame, words_data: DataFrame, thresholds=(3, 6), use_tfidf: bool = True) -> List[str]:
        """Create stopwords for a given corpus (based on tfidf)

        Args:
            thresholds (tuple, optional): threshold for removing rare and common words. Defaults to (3, 8).
            use_tfidf (bool, optional): use tfidf to calculate stopwords. Defaults to True.

        Returns:
            List[str]: custom tf-idf stopwords 
        """
        if not use_tfidf:
            return stopwords.words('russian')

        # Считаем частотности
        count_vectorizer = CountVectorizer(inputCol='words', outputCol='raw_features',
                                           vocabSize=100_000, minDF=10)
        count_model = count_vectorizer.fit(words_data)
        count_data = count_model.transform(words_data)
        # Получаем tf-idf
        idf_model = self.idf.fit(count_data)
        idfs_values = idf_model.idf.values
        rare_words = idfs_values < thresholds[0]
        common_words = idfs_values > thresholds[1]
        mask = np.logical_or(common_words, rare_words)
        stop_words = np.array(count_model.vocabulary)[mask]
        return stop_words

    @staticmethod
    def stemUDF(text: List[str]):
        """NLTK SnowballStemmer wrapper

        Args:
            text (List[str]): one document from corpus

        Returns:
            List[str]: one stemmed document
        """
        stemmer = SnowballStemmer('russian')
        stemmed_list = []
        for word in text:
            stemmed_list.append(stemmer.stem(word))
        return stemmed_list
