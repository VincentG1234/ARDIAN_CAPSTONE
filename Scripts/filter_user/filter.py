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

def regex_replace_company_name_test(df:pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        if pd.notna(row["NAME"]):
            pattern = re.escape(str(row["NAME"]))
            df.at[index, "BUSINESS_DESCRIPTION"] = re.sub(pattern, "The company", str(row["BUSINESS_DESCRIPTION"]), flags=re.IGNORECASE)
    return df