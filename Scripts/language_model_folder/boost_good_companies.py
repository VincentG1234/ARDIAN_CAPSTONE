import pandas as pd
import numpy as np
import re

def index_companies_with_good_FTE(df: pd.DataFrame, row_buildup: pd.DataFrame) -> pd.Index:
    """
    Filtrer les données en fonction des bornes minimales et maximales saisies par l'utilisateur.
    Retourner les index des lignes qui respectent la condition.
    """
    if "FTE" not in df.columns:
        print(f"WARNING: La colonne {'FTE'} n'existe pas dans le DataFrame.")
        return pd.Index([])
    elif row_buildup["FTE"].isnull().values.any():
        print(f"WARNING: La colonne {'FTE'} contient des valeurs nulles.")
        return pd.Index([])
    else:
        pass

    value = row_buildup["FTE"].values[0]
    min_ = max(value * 1 / 100, 10)
    max_ = value * 15 / 100

    # Supprimer les lignes avec des valeurs NA dans la colonne FTE
    df_filtered = df.dropna(subset=["FTE"])

    # Retourner les index des lignes qui respectent la condition
    return df_filtered[df_filtered["FTE"].between(min_, max_, inclusive="both")].index

def index_companies_with_good_ownership(df: pd.DataFrame) -> pd.Index:
    """
    Filtrer les données en fonction des bornes minimales et maximales saisies par l'utilisateur.
    Retourner les index des lignes qui respectent la condition.
    """
    if "OWNERSHIP" not in df.columns:
        print(f"WARNING: La colonne {'OWNERSHIP'} n'existe pas dans le DataFrame.")
        return pd.Index([])
    else:
        pass

    good_ownership = ['private', 'ventureCapital', 'minority', 'regular']

    # Supprimer les lignes avec des valeurs NA dans la colonne OWNERSHIP
    df_filtered = df.dropna(subset=["OWNERSHIP"])

    # Retourner les index des lignes qui respectent la condition
    return df_filtered[df_filtered["OWNERSHIP"].isin(good_ownership)].index