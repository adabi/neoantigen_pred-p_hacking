import pandas as pd
from lifelines import KaplanMeierFitter

#load the raw patients xlsx file
df_patients = pd.read_excel('./data/nature24473_MOESM5_survival.xlsx', sheet_name=None)
df_patients = pd.concat(df_patients.values(), ignore_index=True)

# check which patients have status 1 (dead) and survival time < 12 months and convert to int
df_patients['year_death'] = ((df_patients['Status'] == 1) & (df_patients['Months'] < 12))
df_patients['year_death'] = df_patients['year_death'].astype(int)


# save dataframe to csv
df_patients.to_csv('./data/processed/patients.csv', index=False)