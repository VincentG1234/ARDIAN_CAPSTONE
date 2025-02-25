import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PCA_functions import reduce_dimension_pca
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode(text, model, tokenizer):
    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return embedding



def get_embedding_matrix(df, model, tokenizer):
    """Retourne une matrice d'embedding BERT pour une liste de textes."""
    
    embedding_matrix = []
    for text in tqdm(df["BUSINESS_DESCRIPTION"], desc="Processing embeddings"):
        embedding_matrix.append(encode(text, model, tokenizer))

    return np.vstack(embedding_matrix)



def get_top_n_similar_companies(df, reduced_embeddings, company_index, top_n=10):
    """Retourne les 10 entreprises les plus similaires à une entreprise donnée."""

    # Calcul de la similarité cosinus entre l'embedding cible et tous les autres
    similarities = cosine_similarity(reduced_embeddings[company_index].reshape(1, -1), reduced_embeddings)[0]

    # Exclure le vecteur lui-même
    similarities[company_index] = -1  

    # Trouver les indices des 10 valeurs les plus élevées (hors soi-même)
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]  # Tri décroissant

    print(f"Top {top_n} des entreprises les plus similaires à {df.loc[company_index, 'NAME']} :\n {df.loc[top_n_indices, 'NAME']}")


def pipeline_model(df, index, model, tokenizer, top_n=10):
    """Pipeline complet pour obtenir les 10 entreprises les plus similaires à une entreprise donnée."""

    # # Créer une copie temporaire du DataFrame pour éviter toute modification permanente
    # temp_df = df.copy()

    # # Vérifier si l'utilisateur a fourni une description personnalisée
    # if custom_description:
    #     temp_df.loc[index, "BUSINESS_DESCRIPTION"] = custom_description  # Remplacement temporaire

    # Calcul de l'embedding BERT sur la copie
    embedding_matrix = get_embedding_matrix(df, model, tokenizer)

    # Réduction de la dimension
    reduced_embeddings = reduce_dimension_pca(embedding_matrix, variance_threshold=0.90)

    # Obtenir les 10 entreprises les plus similaires
    get_top_n_similar_companies(df, reduced_embeddings, index, top_n=top_n)
