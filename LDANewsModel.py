from typing import List

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import (IDF, CountVectorizer, RegexTokenizer,
                                StopWordsRemover)
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType


class LDANewsModel:

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

    def fit(self, num_topics: int = 5, max_iterations: int = 20) -> LDA:
        # TODO: write documentation
        data = self.preprocess()

        # final TF matrix
        # TODO: optimize usage of CountVectorizer
        count_vectorizer = CountVectorizer(inputCol='stemmed_words', outputCol='tf',
                                           vocabSize=100_000, minDF=10)
        count_model = count_vectorizer.fit(data)
        tf_data = count_model.transform(data)

        # LDA
        lda_model = LDA(featuresCol='tf', k=num_topics,
                        maxIter=max_iterations)
        result = lda_model.fit(tf_data[['id', 'tf']])
        return result, count_model

    def preprocess(self) -> DataFrame:
        """Preprocess corpus (tokenization, stopwords, stemming)

        Returns:
            DataFrame: preprocessed spark DataFrame
        """
        # Removing stopwords
        stop_words = self.calculate_stopwords()
        remover = StopWordsRemover(
            inputCol='words', outputCol='cleared_words', stopWords=stop_words)
        cleared_texts = remover.transform(self._words_data)
        # Stemming
        stemmingUDF = udf(self.stemUDF, ArrayType(StringType()))
        stemmed_text = cleared_texts.select(col('id'), stemmingUDF(
            col('cleared_words')).alias('stemmed_words'))
        return stemmed_text

    def calculate_stopwords(self, thresholds=(3, 6), use_tfidf: bool = True) -> List[str]:
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
        count_model = count_vectorizer.fit(self._words_data)
        count_data = count_model.transform(self._words_data)
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
