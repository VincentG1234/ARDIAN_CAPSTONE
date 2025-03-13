import pandas as pd 
import numpy as np
import re


def regex_replace_company_name(df):
    for index, row in df.iterrows():
        if pd.notna(row["NAME"]):
            pattern = re.escape(str(row["NAME"]))
            df.at[index, "BUSINESS_DESCRIPTION"] = re.sub(pattern, "The company", str(row["BUSINESS_DESCRIPTION"]), flags=re.IGNORECASE)
    return df

def regex_on_tags(df):
    """
    Remove special characters from TAGS column.
    """
    
    df["TAGS"] = df["TAGS"].apply(lambda tags: re.sub(r'[\[\]\r\n",]', '', str(tags)).strip())
    return df

