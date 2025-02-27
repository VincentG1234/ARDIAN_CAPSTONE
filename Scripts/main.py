import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Scripts.language_model_folder.language_model import pipeline_model
from Scripts.filter_user.ask_user import ask_all, ask_company, ask_keywords, ask_whether_bound, ask_filter
from Scripts.filter_user.filter import set_bounds

def searchmodel_main(df, model=None, tokenizer=None, top_n=10 ) -> pd.DataFrame:
    
    # Filtration pour récupérer l'entreprise mère
    df_buildup = df.copy()

    print("Entrez le pays de l'entreprise mère. ex: FR, DE, UK, US... \n")
    df_buildup = ask_filter(df_buildup, column_name="REGION", prompt="Entrez le Pays de l'entreprise mère: ")

    # Sélection de l'entreprise à partir de laquelle on souhaite rechercher des entreprises similaires
    row_buildup_firm = ask_company(df_buildup).iloc[0:1]
    
    # Suppression de la ligne de l'entreprise choisie
    index = row_buildup_firm.index
    df = df.drop(index)

    # Filtrage des données en fonction des bornes minimales et maximales sur le FTE
    if ask_whether_bound().lower() == "y":
        df = set_bounds(df, row_buildup_firm, column_name="FTE", min_bound=20, max_bound=50)
    elif ask_whether_bound().lower() == "n":
        pass
    else:
        print("Valeur non reconnue. Veuillez entrer 'y' ou 'n'.")
        return None
    
    # Filtrage des données selon les préférences de l'utilisateur
    df = ask_all(df)


    # Enrichissement de la description avec des mots-clés saisis par l'utilisateur
    buildup_firm_description = row_buildup_firm["BUSINESS_DESCRIPTION"].values[0]
    buildup_firm_description = ask_keywords(buildup_firm_description)

    # reset index avant le pipeline
    df.reset_index(drop=True, inplace=True)
    
    # Appliquer le modèle BERT avec la nouvelle description enrichie
    pipeline_model(df, buildup_firm_description, model=model, tokenizer=tokenizer, top_n=top_n)




