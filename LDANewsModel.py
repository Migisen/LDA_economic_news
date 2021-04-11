from typing import List

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import (CountVectorizerModel, IDF, CountVectorizer, RegexTokenizer,
                                StopWordsRemover)
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
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
        self.__trained_model: LDAModel
        self.__fit_data: DataFrame
        self.__cv_fit_data: CountVectorizerModel

    def fit(self, num_topics: int = 5, max_iterations: int = 100, thresholds: tuple = (3, 6)) -> tuple[LDAModel, CountVectorizerModel]:
        # TODO: write documentation
        self.__fit_data, self.__cv_fit_data = self.preprocess(
            data=self.news, thresholds=thresholds)
        self.__fit_data.repartition(12).cache()
        # LDA
        lda_model = LDA(featuresCol='tf', k=num_topics, maxIter=max_iterations)
        self.__trained_model = lda_model.fit(self.__fit_data.select(['id', 'tf']))
        return self.__trained_model, self.__cv_fit_data

    def fit_predict(self, num_topics: int = 5, max_iterations: int = 100, thresholds: tuple = (3, 6)):
        """Same as calling fit and then predict on the same data but faster

        Args:
            num_topics (int, optional): number of desired topics. Defaults to 5.
            max_iterations (int, optional): max iterations for LDA. Defaults to 100.
            thresholds (tuple, optional): idf coefficients for stopwords generator. Defaults to (3, 6).

        Returns:
            tuple[LDAModel, CountVectorizerModel, DataFrame]: model and it's predictions
        """
        self.fit(num_topics, max_iterations, thresholds)
        preidction = self.__trained_model.transform(self.__fit_data)
        return self.__trained_model, self.__cv_fit_data, preidction

    def predict(self, data: DataFrame):
        """Get predictions on new data

        Args:
            data (DataFrame): text corpus

        Returns:
            [type]: [description]
        """
        assert self.__trained_model is not None, 'Train model first'
        preprocessed_data, cv_model = self.preprocess(data)
        print('Done!')
        prediction = self.__trained_model.transform(preprocessed_data)
        return prediction, cv_model

    def preprocess(self, data: DataFrame, thresholds=(3, 6)) -> tuple[DataFrame, CountVectorizerModel]:
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

        # Calculate frequencies
        count_model = self.count_vectorizer.fit(words_data)
        count_data = count_model.transform(words_data)
        # Calculating idf
        idf_model = self.idf.fit(count_data)
        idfs_values = idf_model.idf.toArray()
        mask = np.logical_or(
            idfs_values < thresholds[0], idfs_values > thresholds[1])
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
