import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# load the dataframe
df = pd.read_csv('./data/processed/embeddings.csv')


# x is all the embeddings
x = df.drop(columns=['Sample', 'year_death', 'Months', 'Status'])

# y is the year_death column
y = df['year_death'].values

# fit a random forest model with x and y
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
rf.fit(x, y)

# sort the features by importance
feature_importances = rf.feature_importances_
feature_names = x.columns
feature_importances = np.array(feature_importances)
sort_idx = np.argsort(feature_importances)
feature_importances = feature_importances[sort_idx]
feature_names = feature_names[sort_idx]

# Select the top 20 most important features
top_features = feature_names[-200:]

x_train, x_test, y_train, y_test = train_test_split(x, df, test_size=0.2)

p_values = []

# test each feature with a logistic regression
for feature in top_features:
    print(feature)
    x_train_feature = x_train[feature].values.reshape(-1, 1)
    x_test_feature = x_test[feature].values.reshape(-1, 1)
    log_reg = linear_model.LogisticRegression(max_iter=1000)
    log_reg.fit(x_train_feature, y_train['year_death'])
    y_pred = (log_reg.predict(x_test_feature))
    y_test_yes = y_test[y_pred == 1]
    y_test_no = y_test[y_pred == 0]

    # calculate log rank p value
    results = logrank_test(y_test_yes['Months'], y_test_no['Months'], y_test_yes['Status'], y_test_no['Status'], alpha=.99)

    p_value = results.p_value

    if p_value < 0.0005:
        break

# calculate log rank p value
results = logrank_test(y_test_yes['Months'], y_test_no['Months'], y_test_yes['Status'], y_test_no['Status'], alpha=.99)

p_value = results.p_value

print('log rank p value:')
print(p_value)


# plot kaplan meier curves for train for each of the two groups
kmf = KaplanMeierFitter()
kmf.fit(y_test_yes['Months'], y_test_yes['Status'].astype(bool), label='yes')
ax = kmf.plot()
kmf.fit(y_test_no['Months'], y_test_no['Status'].astype(bool), label='no')
kmf.plot(ax=ax)

# save the figure
plt.savefig('./data/processed/kaplan_meier.png')

