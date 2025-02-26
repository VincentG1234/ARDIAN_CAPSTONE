import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def compute_optimal_pca_components(embedding_matrix, variance_threshold=0.90):
    """
    Calcule le nombre optimal de dimensions à conserver en fonction du seuil de variance.

    Parameters:
        embedding_matrix (np.array): Matrice des embeddings (nb_samples, dim_originale).
        variance_threshold (float): Seuil de variance à conserver (ex: 0.95 pour 95%).

    Returns:
        int: Nombre optimal de dimensions à conserver.
    """
    pca = PCA()
    pca.fit(embedding_matrix)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Trouver le nombre minimal de dimensions qui dépasse le seuil
    optimal_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1
    print(f"Nombre de dimensions retenues : {optimal_components} (pour {variance_threshold*100:.1f}% de variance)")

    return optimal_components

def apply_pca(embedding_matrix, n_components):
    """
    Applique la réduction de dimension PCA avec un nombre donné de composantes.

    Parameters:
        embedding_matrix (np.array): Matrice des embeddings.
        n_components (int): Nombre de dimensions après réduction.

    Returns:
        np.array: Matrice d'embedding réduite.
        PCA: Modèle PCA entraîné.
    """
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(embedding_matrix)
    return reduced_matrix, pca

def reduce_dimension_pca(embedding_matrix, variance_threshold=0.85):
    """
    Réduit la dimension de la matrice d'embedding en conservant un certain seuil de variance.

    Parameters:
        embedding_matrix (np.array): Matrice des embeddings.
        variance_threshold (float): Seuil de variance à conserver.
        plot_variance (bool): Afficher le graphique de variance cumulée.

    Returns:
        np.array: Matrice d'embedding réduite.
        PCA: Modèle PCA entraîné.
    """
    print("Calcul de la réduction de dimension avec PCA...")

    # Étape 1 : Trouver le nombre optimal de dimensions
    optimal_components = compute_optimal_pca_components(embedding_matrix, variance_threshold)

    # Étape 2 : Appliquer PCA avec le nombre optimal de dimensions
    reduced_matrix, pca = apply_pca(embedding_matrix, optimal_components)

    return reduced_matrix, pca

