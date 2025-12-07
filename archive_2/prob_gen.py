import pandas as pd
import io
import requests
import country_converter as coco


df = pd.read_csv('equities.csv')
# Remove companies missing values in relevant columns
relevant_columns = ['country', 'sector', 'market_cap']
df = df.dropna(subset=relevant_columns).copy()
# Map country to continent
cc = coco.CountryConverter()
df['continent'] = cc.convert(
    names=df['country'].tolist(), to='continent_7', not_found="na")
df = df[df['continent'] != 'na']
# Calculate joint proportions
ret = df.groupby(
    ['continent', 'sector', 'market_cap']).size() / len(df)
ret.name = "probability"
ret.to_csv("prob.csv")
print("Successfully saved to 'prob.csv'")
