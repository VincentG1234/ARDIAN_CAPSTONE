import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from language_model_folder.language_model import pipeline_model
from filter_user.ask_user import ask_all, ask_company, ask_keywords

def searchmodel_main(df, model=None, tokenizer=None, top_n=10 ) -> pd.DataFrame:
    
    # Filtrage des données selon les préférences de l'utilisateur
    df = ask_all(df)

    # Selection de l'entreprise
    index = ask_company(df)

    # print(index)
    # # Récupérer la description de l'entreprise choisie

    # if ask_keywords() is not None:
    #     add_to_description_keywords = "KEY WORDS; ask_keywords()."
    # else:
    #     add_to_description_keywords = ""

    # Enrichissement de la description avec des mots-clés saisis par l'utilisateur
    df.loc[index, "BUSINESS_DESCRIPTION"] = ask_keywords(df.loc[index, "BUSINESS_DESCRIPTION"])

    # Ajouter des mots-clés pour guider la recherche
    # chosen_description = add_keywords_to_description(chosen_description)

    # Appliquer le modèle BERT avec la nouvelle description enrichie
    pipeline_model(df, index, model=model, tokenizer=tokenizer, top_n=top_n)




