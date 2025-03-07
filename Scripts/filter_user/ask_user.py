from typing import Union, List
import ast
import pandas as pd
from fuzzywuzzy import process
import sys
import os
import importlib
import time
from Scripts.filter_user.filter import set_bounds

#  Fonction universelle pour récupérer l'input utilisateur
def get_user_input(message: str) -> str:
    """
    Récupère l'entrée utilisateur via `input()`.
    Fonctionne aussi bien dans un Notebook Databricks qu'en local.
    """
    return input(message).strip()


def ask_filter(df: pd.DataFrame, column_name: str, prompt: str) -> pd.DataFrame:
    """
    Demande à l'utilisateur de choisir une ou plusieurs valeurs pour filtrer un DataFrame.

    Parameters:
        df (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne à filtrer (ex: "REGION", "SECTOR", "SUBSECTOR").
        prompt (str): Le message à afficher pour demander une entrée utilisateur.

    Returns:
        pd.DataFrame: Le DataFrame filtré.
    """
    available_values = df[column_name].unique()

    if len(available_values) < 30:
        message = (
        f"Choisissez une valeur parmi celles disponibles pour {column_name} :\n{available_values}\n"
        "Exemple : 'FR' pour un seul élément, ['FR', 'DE'] pour plusieurs, ou 'all' pour tous.\n"
    )
    else:
        message = (
            f"Choisissez une valeur pour {column_name}. Exemple de valeur disponible:\n{available_values[:15]} \n"
            "Exemple : 'FR' pour un seul élément, ['FR', 'DE'] pour plusieurs, ou 'all' pour tous.\n"
        )

    print(message)

    time.sleep(1)
    
    try:
        user_input = get_user_input(prompt)

        if user_input.lower() == "all":
            return df

        if user_input.startswith("[") and user_input.endswith("]"):
            try:
                values_list = ast.literal_eval(user_input)
                if isinstance(values_list, list) and all(isinstance(v, str) for v in values_list):
                    return df[df[column_name].isin(values_list)]
                else:
                    print("Erreur : La liste doit contenir uniquement des chaînes de caractères.")
                    return None
            except (SyntaxError, ValueError):
                print("Erreur : Format invalide pour la liste.")
                return None

        return df[df[column_name] == user_input]

    except Exception as e:
        print(f"Erreur lors de la saisie : {e}")
        return None


def ask_company(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demande à l'utilisateur de choisir une entreprise après le filtrage.
    """
  
    names = df["NAME"][:10]

    print(f"Choisissez une entreprise, voici quelques exemples :\n{names}")

    try:
        company = get_user_input("Votre choix : ")

        if company not in df["NAME"].unique():
            print("Entreprise non trouvée. Suggestions proches :")
            suggestions = process.extract(company, df["NAME"].unique(), limit=5)
            for suggestion in suggestions:
                print(f"- {suggestion[0]} (score: {suggestion[1]})")

            print("Veuillez réessayer:")
            ask_company(df)
        else:
            return df[df["NAME"] == company]

    except Exception as e:
        print(f"Erreur lors de la saisie : {e}")
        return None


def ask_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demande tous les choix : pays, secteur, sous-secteur et entreprise.
    """
    df = ask_filter(df, "REGION", "Votre choix pour le pays : ")
    if df is None or df.empty:
        print("Erreur lors du choix du pays.")
        return 

    df = ask_filter(df, "SECTOR", "Votre choix pour le secteur : ")
    if df is None or df.empty:
        print("Erreur lors du choix du secteur.")
        return

    df = ask_filter(df, "SUBSECTOR", "Votre choix pour le sous-secteur : ")
    if df is None or df.empty:
        print("Erreur lors du choix du sous-secteur.")
        return

    selected_region = df["REGION"].unique()
    selected_sector = df["SECTOR"].unique()
    selected_subsector = df["SUBSECTOR"].unique()

    print(f"Vous avez choisi : Pays = {selected_region}, Secteur = {selected_sector}, Sous-secteur = {selected_subsector}")
    return df

def ask_keywords(description: str, firm_tags: list = None) -> str:
    """
    Demande à l'utilisateur s'il souhaite ajouter des mots-clés, 
    et ajoute les mots-clés de l'entreprise mère avec un poids plus important.

    Parameters:
        description (str): Description actuelle de l'entreprise.
        firm_tags (list): Liste des mots-clés de l'entreprise mère.

    Returns:
        str: Nouvelle description enrichie.
    """
    if firm_tags:
        print(f"L'entreprise mère contient ces mots-clés : {', '.join(firm_tags)}")
        use_firm_tags = get_user_input("Voulez-vous utiliser ces mots-clés ? (y/n) : ").strip().lower()
        
        if use_firm_tags == "y":
            weighted_tags = (", ".join(firm_tags) + " ") * 3
            return f"{weighted_tags.strip()}. {description}"

    print("Entrez des mots-clés supplémentaires (séparés par des virgules) pour affiner la recherche : ")
    user_input = get_user_input("Mots-clés: ").strip()

    if not user_input:  
        print("Aucun mot-clé ajouté.")
        return description

    keywords = [kw.strip().lower() for kw in user_input.split(",") if kw.strip()]
    formatted_keywords = ", ".join(keywords)

    weighted_keywords = (formatted_keywords + " ") * 3
    enriched_description = f"{weighted_keywords.strip()}. {description}"

    return enriched_description

def ask_whether_bound() -> pd.DataFrame:
    
    print("Voulez-vous appliquer un filtre sur les valeurs de la colonne FTE/REVENUE/EBITDA ?")
    user_input = get_user_input("y/n: ")
    return user_input
