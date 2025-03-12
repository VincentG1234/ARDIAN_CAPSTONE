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



def get_embedding_matrix(df, model, tokenizer):
    """Retourne une matrice d'embedding BERT pour une liste de textes."""
    
    embedding_matrix = []
    for text in tqdm(df["BUSINESS_DESCRIPTION"], desc="Processing embeddings"):
        embedding_matrix.append(encode(text, model, tokenizer))

    return np.vstack(embedding_matrix)



def get_top_n_similar_companies(df, reduced_embeddings, buildup_description_encoded_reduced, top_n=10):
    """Retourne les 10 entreprises les plus similaires à une entreprise donnée."""

    # Calcul de la similarité cosinus entre l'embedding cible et tous les autres
    similarities = cosine_similarity(buildup_description_encoded_reduced.reshape(1, -1), reduced_embeddings)[0]

    # Trouver les indices des 10 valeurs les plus élevées (hors soi-même)
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]  # Tri décroissant
    pd.set_option('display.max_rows', None)  

    print(f"Top {top_n} des entreprises les plus similaires:\n {df.loc[top_n_indices, 'NAME']}")
    result_df = df.loc[top_n_indices, ['NAME', 'TAGS', 'BUSINESS_DESCRIPTION']].assign(Similarity=similarities[top_n_indices])
    print(similarities[top_n_indices])
    print(similarities.shape)
    result_df['Similarity'] = similarities[top_n_indices]
    return result_df

def scoring_topn(firm, results, targets, top_n, data):
    """Retourne le nombre de build ups présents dans les ntop résultats du modèle"""
    top_index = np.argsort(results)[-top_n:][::-1]
    top_results = data.loc[top_index, 'NAME'].tolist()
    if firm in targets.keys():
        prensent = 0
        for target in targets[firm]:
            if target in top_results:
                prensent += 1
    
        print(f"Le score du modèle pour {firm} est de {prensent}/{len(targets[firm])}, pour le top {len(top_results)} résultats")
    else:
        print(f"l'entreprise {firm} n'est pas dans la liste des entreprises mères")

def scoring_pos(firm, results, targets, data):
    """"Retourne la position des builds ups dans la liste des résultats """
    if firm in targets.keys():
        present = []
        for target in targets[firm]:
            if target in results:
                print(f"Le build up {target} est classé à la position {results.index(target)} sur {len(results)} entreprises")
                present.append(results.index(target))
            else:
                print(f"Le build up {target} n'est pas présent dans les résultats pour les filtres appliqués")

        #plots a bar plot of every similarity score, with the indexes of present in a different color
        sns.set_theme(style="whitegrid")
        sns.barplot(x=data.loc[results, 'NAME'].tolist(), y = results, palette="Blues") 
        plt.xticks(rotation=90)
        plt.show()

    

def pipeline_model(df, buildup_description, model, tokenizer, top_n=10):
    """Pipeline complet pour obtenir les 10 entreprises les plus similaires à une entreprise donnée."""

    # # Créer une copie temporaire du DataFrame pour éviter toute modification permanente
    # temp_df = df.copy()

    # # Vérifier si l'utilisateur a fourni une description personnalisée
    # if custom_description:
    #     temp_df.loc[index, "BUSINESS_DESCRIPTION"] = custom_description  # Remplacement temporaire

    # Calcul de l'embedding BERT sur la copie
    embedding_matrix = get_embedding_matrix(df, model, tokenizer)

    buildup_description_encoded = encode(buildup_description, model, tokenizer)

    # Réduction de la dimension
    reduced_embeddings, pca = reduce_dimension_pca(embedding_matrix, variance_threshold=0.85)
    buildup_description_encoded_reduced = pca.transform(buildup_description_encoded)

    # Obtenir les 10 entreprises les plus similaires
    df_result = get_top_n_similar_companies(df, reduced_embeddings, buildup_description_encoded_reduced, top_n=top_n)
    
    return df_result
