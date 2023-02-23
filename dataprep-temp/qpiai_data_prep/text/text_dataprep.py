import json
import os
import re
import string
import unicodedata

import contractions
import inflect
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from validator_collection import checkers

from qpiai_data_prep.text.helpers.dataset import *
from qpiai_data_prep.text.helpers.utils import *
from temp.db_info import db

def text_dataprep(dataset, target_device, num_device, data_prepcmd, clmn, **kwargs):
    db.update_progress(progress=5)
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    db.update_progress(progress=25)
    f_le, file_name = text_download(dataset)
    db.update_progress(progress=45)
    if file_name.split(".")[-1] == "txt":
        if data_prepcmd == "word2vec":
            db.update_progress(progress=55)
            text = f_le.read()
            text = text_ops.strip_html(text)
            text = text_ops.remove_between_square_brackets(text)
            text = contractions.fix(text)
            words = nltk.word_tokenize(text)
            words = text_ops.remove_non_ascii(words)
            db.update_progress(progress=70)
            words = text_ops.to_lowercase(words)
            words = text_ops.remove_punctuation(words)
            words = text_ops.replace_numbers(words)
            db.update_progress(progress=85)
            words = text_ops.remove_stopwords(words)

            def stem_and_lemmatize(words):
                stems = text_ops.stem_words(words)
                lemmas = text_ops.lemmatize_verbs(words)
                return stems, lemmas

            stems, lemmas = stem_and_lemmatize(words)
            text_ops.word2vec([stems])
            filename = "stems_file"
            with open(filename, "w") as file:
                for s in stems:
                    file.write(s)
                    file.write("\n")
            file.close()
            filename1 = "lemmas_file"
            with open(filename1, "w") as file:
                for l in lemmas:
                    file.write(l)
                    file.write("\n")
            file.close()
            checkpoint = dict(
                {
                    "dataPrepOutput": [
                        os.path.abspath(filename),
                        os.path.abspath(filename1),
                        os.path.abspath("word2vec.model"),
                    ]
                }
            )
        elif data_prepcmd == "text_to_speech":
            db.update_progress(progress=65)
            mytext = f_le.read()
            text_ops.text_to_speech(mytext)
            db.update_progress(progress=75)
            checkpoint = dict({"dataPrepOutput": [os.path.abspath("sample.mp3")]})
        else:
            return "Please provide the Data prep command"
    else:
        f_le_df = pd.read_csv(file_name)
        db.update_progress(progress=65)
        if data_prepcmd == "tf_idf":
            text_ops.tf_idf(f_le_df[clmn])
            db.update_progress(progress=85)
            checkpoint = dict({"dataPrepOutput": [os.path.abspath("tf_idf.csv")]})
        elif data_prepcmd == "bag_of_words":
            text_ops.bag_of_words(f_le_df[clmn])
            db.update_progress(progress=85)
            checkpoint = dict({"dataPrepOutput": [os.path.abspath("bow.csv")]})
        elif data_prepcmd == "word2vec":
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: text_ops.strip_html(text))
            f_le_df[clmn] = f_le_df[clmn].apply(
                lambda text: text_ops.remove_between_square_brackets(text)
            )
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: contractions.fix(text))
            f_le_df[clmn] = f_le_df[clmn].apply(
                lambda text: text_ops.rem_non_ascii(text)
            )
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: text_ops.to_lowcase(text))
            f_le_df[clmn] = f_le_df[clmn].apply(
                lambda text: text_ops.rem_punctuation(text)
            )
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: text_ops.rep_numbers(text))
            f_le_df[clmn] = f_le_df[clmn].apply(
                lambda text: text_ops.rem_stopwords(text)
            )
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: text_ops.stm_words(text))
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: text_ops.lemm_verbs(text))
            f_le_df[clmn] = f_le_df[clmn].apply(lambda text: nltk.word_tokenize(text))
            f_le_df.to_csv("output_text.csv")
            db.update_progress(progress=85)
            text_ops.word2vec(f_le_df[clmn])
            checkpoint = dict(
                {
                    "dataPrepOutput": [
                        os.path.abspath("output_text.csv"),
                        os.path.abspath("word2vec.model"),
                    ]
                }
            )
        else:
            return "Please provide the Data prep command"

    # checkpoint = dict({'dataPrepOutput': [os.path.abspath(filename),os.path.abspath(filename1)]})
    print(json.dumps(checkpoint))
    return checkpoint
    # os.path.abspath(filename),os.path.abspath(filename1)


# text_dataprep('https://qpiaidataset.s3.amazonaws.com/Corona_NLP_test.csv', 'cpu', 1, data_prepcmd='tf_idf', clmn='OriginalTweet')
# text_dataprep('https://dataprepfiles.s3.amazonaws.com/new.txt', 'cpu', 1, data_prepcmd='word2vec', clmn=None)
