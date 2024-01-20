import pandas as pd # load &manipulate and for one-hit encoding
import numpy as np # calculate the mean & standard deviation
import xgboost as xgb # xgboost stuff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # split data into trainning & testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # scoring during
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix #creates a confusion matrix
from sklearn.metrics import pair_confusion_matrix # draws a confusion matrix

df = pd.read_csv('skrip.csv')
df.drop(['prediksi awal', 'residual-1'],
        axis=1, inplace=True) #axis=1 to remove coloums, axis=0 to remove rows
print(df)

# penambahan fitur pendukung
#df['Daily_Return'] = df['Close'].pct_change()
df['MA_20'] = df['Close'].rolling(20).mean()
df['High_low_Diff'] = df['High'] - df['Low']
print(df.head())
#print(df.head(25)) #menampilkan 5(default) data di terminal
df.dtypes #menampilkan tipe data setiap kolom

#pisahkan data menjadi fitur (x) & label (y)
X = df.drop('Close', axis=1).copy()
#print(X.head(25))
y = df['Close'].copy()
#print(y.head(25))

#split data: 80% data pelatihan & 20% pengujian
X_train, X_valid,y_train, y_valid = train_test_split(X, y, train_size=0.8,
                                                     test_size=0.2, random_state=0)

low_cardinality_cols = [cname for cname in X_train.coloums if X_train[cname].nunique() < 10 and 
                        X_train[cname].dtypes == "object"]

#modul xgboost klasifikasi fitu(X)
#clf_xgb = xgb.XGBClassifier(objective='binary:logistic', n_estimators=2, max_depth=2, learning_rate=1, seed=42)
#clf_xgb.fit(X_train, y_train)

from xgboost import XGBRegressor

rg_xgb = XGBRegressor(random_state=0)
rg_xgb.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error

predict1 = rg_xgb.predict(X_valid)

mae_1 = mean_absolute_error(predict1, y_valid)
print(mae_1)