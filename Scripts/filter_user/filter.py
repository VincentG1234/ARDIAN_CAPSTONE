import pandas as pd 
import numpy as np
import re

def set_bounds(df: pd.DataFrame, row_buildup: pd.DataFrame, column_name: str = "FTE", min_bound=20, max_bound = 50) -> pd.DataFrame:
    """
    Filtrer les données en fonction des bornes minimales et maximales saisies par l'utilisateur.
    """
    if column_name not in df.columns:
        print(f"WARNING: La colonne {column_name} n'existe pas dans le DataFrame.")
        return None
    elif row_buildup[column_name].isnull().values.any():
        print(f"WARNING: La colonne {column_name} contient des valeurs nulles.")
        return None
    else:
        pass

    value = row_buildup[column_name].values[0]
    min_ = value * min_bound / 100
    max_ = value * max_bound / 100 

    return df[df[column_name].between(min_, max_, inclusive="both")]

def regex_replace_company_name(df:pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        if pd.notna(row["NAME"]):
            pattern = re.escape(str(row["NAME"]))
            df.at[index, "BUSINESS_DESCRIPTION"] = re.sub(pattern, "The company", str(row["BUSINESS_DESCRIPTION"]), flags=re.IGNORECASE)
    return df

def adjust_business_description_with_tags(df, row_buildup_firm):
    """
    Ajoute les mots-clés de l'entreprise mère dans la description des autres entreprises
    en leur attribuant un poids plus élevé.
    
    Parameters:
        df (pd.DataFrame): DataFrame contenant toutes les entreprises.
        row_buildup_firm (pd.DataFrame): Entreprise mère sélectionnée.

    Returns:
        pd.DataFrame: DataFrame mis à jour avec les descriptions enrichies.
    """
    if "TAGS" not in df.columns or "TAGS" not in row_buildup_firm.columns:
        print("Colonne TAGS non trouvée dans le DataFrame.")
        return df

    # Extraire les tags de l'entreprise mère
    firm_tags = row_buildup_firm["TAGS"].values[0]
    
    if not isinstance(firm_tags, str):
        print("Les tags de l'entreprise mère sont invalides ou absents.")
        return df

    try:
        firm_tags_list = eval(firm_tags) if isinstance(firm_tags, str) else firm_tags
        firm_tags_list = [tag.strip().lower() for tag in firm_tags_list]
    except:
        print("Impossible de lire les tags de l'entreprise mère.")
        return df

    def enrich_description(description, tags):
        """ Ajoute les tags avec un poids plus élevé à la description. """
        weighted_tags = (", ".join(tags) + " ") * 3  # Répétition pour augmenter l'importance
        return f"{weighted_tags.strip()}. {description}"

    # Appliquer l'enrichissement à toutes les descriptions
    df["BUSINESS_DESCRIPTION"] = df["BUSINESS_DESCRIPTION"].apply(lambda desc: enrich_description(desc, firm_tags_list))
    
    return df
