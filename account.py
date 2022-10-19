import pandas as pd
import numpy as np

df=pd.read_csv("account.csv")

company = df['name'] == "Carroll PLC"
print(df[company])
