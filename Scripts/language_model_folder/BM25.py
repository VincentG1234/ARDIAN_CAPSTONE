import re
import pandas as pd
import spacy
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

def regex_on_tags(df):
    """
    Remove special characters from TAGS column.
    """
    
    df["TAGS"] = df["TAGS"].apply(lambda tags: re.sub(r'[\[\]\r\n",]', '', str(tags)).strip())


def merge_tags_description(df):
    """
    Merge tags and description columns into one column.
    """
    # 1. Fusionner BUSINESS_DESCRIPTION et TAGS
    df["COMBINED_TEXT"] = df["BUSINESS_DESCRIPTION"] + " " + df["TAGS"] 




# Fonction de prétraitement du texte
def preprocess_text(text):
    # Supprimer les caractères parasites (crochets, guillemets, virgules, sauts de ligne)
    text = re.sub(r'\s+', ' ', text).strip()

    # Passer le texte dans SpaCy
    doc = nlp(text.lower())

    # Lemmatiser et supprimer les stopwords anglais
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]

    return " ".join(tokens)

def load_stopword():
    stop_words = set(stopwords.words("english"))
    nlp = spacy.load("en_core_web_md")
    return stop_words, nlp


def BM25_main(df, query):
        

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query)
    return scores

