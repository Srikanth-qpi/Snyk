import os
import re
import string
import unicodedata

import contractions
import inflect
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from gtts import gTTS
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class text_ops:
    def __init__(self):
        pass

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_between_square_brackets(text):
        return re.sub("\[[^]]*\]", "", text)

    def remove_non_ascii(words):
        new_words = []
        for word in words:
            new_word = (
                unicodedata.normalize("NFKD", word)
                .encode("ascii", "ignore")
                .decode("utf-8", "ignore")
            )
            new_words.append(new_word)
        return new_words

    def rem_non_ascii(text):
        return " ".join(
            [
                unicodedata.normalize("NFKD", word)
                .encode("ascii", "ignore")
                .decode("utf-8", "ignore")
                for word in text.split()
            ]
        )

    def to_lowcase(text):
        return text.lower()

    def to_lowercase(words):
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r"[^\w\s]", "", word)
            if new_word != "":
                new_words.append(new_word)
        return new_words

    def rem_punctuation(text):
        return re.sub(r"[^\w\s]", "", text)

    def replace_numbers(words):
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def rep_numbers(text):
        p = inflect.engine()
        if text.isdigit():
            new_word = p.number_to_words(text)
        else:
            new_word = text
        return new_word

    def remove_stopwords(words):
        new_words = []
        for word in words:
            if word not in stopwords.words("english"):
                new_words.append(word)
        return new_words

    def rem_stopwords(text):
        if text not in stopwords.words("english"):
            new_word = text
        return new_word

    def stem_words(words):
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def stm_words(text):
        stemmer = LancasterStemmer()
        return " ".join([stemmer.stem(word) for word in text.split()])

    def lemmatize_verbs(words):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos="v")
            lemmas.append(lemma)
        return lemmas

    def lemm_verbs(text):
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word, pos="v") for word in text.split()])

    def tf_idf(words):
        tf_idf_vec = TfidfVectorizer(
            use_idf=True, ngram_range=(2, 2), stop_words="english", max_features=2000
        )
        tf_idf_vec.fit(words)
        tf_idf_data = tf_idf_vec.transform(words)
        tf_idf_df = pd.DataFrame(
            tf_idf_data.toarray(), columns=tf_idf_vec.get_feature_names()
        )
        return tf_idf_df.to_csv("tf_idf.csv")

    def bag_of_words(words):
        CountVec = CountVectorizer(
            ngram_range=(2, 2), stop_words="english", max_features=2000
        )
        CountVec.fit(words)
        Count_data = CountVec.transform(words)
        cv_df = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names())
        return cv_df.to_csv("bow.csv")

    def text_to_speech(mytext):
        myobj = gTTS(text=mytext, lang="en", slow=False)
        return myobj.save("sample.mp3")

    def word2vec(sentences):
        model = Word2Vec(sentences, min_count=1, window=5, sg=1)
        return model.save("word2vec.model")
