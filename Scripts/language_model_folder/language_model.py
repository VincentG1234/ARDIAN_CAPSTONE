import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from language_model_folder.PCA_functions import reduce_dimension_pca
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



def get_embedding_matrix(df, model, tokenizer, column_name="BUSINESS_DESCRIPTION"):
    """Retourne une matrice d'embedding BERT pour une liste de textes."""
    
    embedding_matrix = []
    for text in tqdm(df[column_name], desc="Processing embeddings " + column_name):
        embedding_matrix.append(encode(text, model, tokenizer))

    return np.vstack(embedding_matrix)

def get_similarity_vect(reduced_embedding, buildup_encoded_reduced):
    """Retourne la similarité cosinus entre un embedding réduit et un autre."""
    return cosine_similarity(buildup_encoded_reduced.reshape(1, -1), reduced_embedding)[0]

def get_top_n_similar_companies(df, similarities, top_n=10):
    """Retourne les 10 entreprises les plus similaires à une entreprise donnée."""

    # Trouver les indices des 10 valeurs les plus élevées (hors soi-même)
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]  # Tri décroissant
    pd.set_option('display.max_rows', None)  

    print(f"Top {top_n} des entreprises les plus similaires:\n {df.loc[top_n_indices, 'NAME']}")
    result_df = df.loc[top_n_indices, ['NAME', 'TAGS', 'BUSINESS_DESCRIPTION']].assign(Similarity=similarities[top_n_indices])
    print(similarities[top_n_indices])
    print(similarities.shape)
    result_df['Similarity'] = similarities[top_n_indices]
    return result_df

    

def pipeline_model(df, buildup_df, model, tokenizer, alpha=0.7, top_n=10):
    """Pipeline complet pour obtenir les 10 entreprises les plus similaires à une entreprise donnée."""

    # # Créer une copie temporaire du DataFrame pour éviter toute modification permanente
    # temp_df = df.copy()

    # # Vérifier si l'utilisateur a fourni une description personnalisée
    # if custom_description:
    #     temp_df.loc[index, "BUSINESS_DESCRIPTION"] = custom_description  # Remplacement temporaire

    ### Calcul de l'embedding sur la BUSINESS DESCRIPTION
    print("Calcul de l'embedding sur la BUSINESS DESCRIPTION")
    embedding_matrix_business_desc = get_embedding_matrix(df, model, tokenizer, column_name="BUSINESS_DESCRIPTION")
    buildup_description_encoded = encode(buildup_df['BUSINESS_DESCRIPTION'].values[0], model, tokenizer)

    # Réduction de la dimension
    reduced_embeddings_business_desc, pca_business_desc = reduce_dimension_pca(embedding_matrix_business_desc, variance_threshold=0.85)
    buildup_description_encoded_reduced = pca_business_desc.transform(buildup_description_encoded)


    ### Calcul de l'embedding sur les TAGS
    print("Calcul de l'embedding sur les TAGS")
    embedding_matrix_tags = get_embedding_matrix(df, model, tokenizer, column_name="TAGS")
    buildup_tags_encoded = encode(buildup_df['TAGS'].values[0], model, tokenizer)
    reduced_embeddings_tags, pca_tags = reduce_dimension_pca(embedding_matrix_tags, variance_threshold=0.95)
    buildup_tags_encoded_reduced = pca_tags.transform(buildup_tags_encoded)

    ### Calcul de la similarité
    sim_vect_desc = get_similarity_vect(reduced_embeddings_business_desc, buildup_description_encoded_reduced)
    sim_vect_tags = get_similarity_vect(reduced_embeddings_tags, buildup_tags_encoded_reduced)

    print("Calcul de la similarité combinée")
    similarities_combined = alpha * sim_vect_desc + (1 - alpha) * sim_vect_tags

    # Obtenir les 10 entreprises les plus similaires
    df_result = get_top_n_similar_companies(df, similarities_combined, top_n=top_n)
    
    return df_result

