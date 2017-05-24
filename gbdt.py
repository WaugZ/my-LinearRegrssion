from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd

train_temp = np.loadtxt('save_train.csv', delimiter=',', skiprows=(1));

X = train_temp[:, 1:-1];
y = train_temp[:, -1:];

clf = GradientBoostingRegressor()
clf.fit(X, y)

# X_cv = train_temp[-10:, 1:-1];
# y_cv = train_temp[-10:, -1:];
# y_predict = clf.predict(X_cv);

test_temp = np.loadtxt('save_test.csv', delimiter=',', skiprows=(1));


X_test = test_temp[:, 1:]
y_predit = clf.predict(X_test)

df = pd.DataFrame(y_predit)
df.to_csv('predict.csv', float_format='%.6f', index_label='Id', header=['reference'])

