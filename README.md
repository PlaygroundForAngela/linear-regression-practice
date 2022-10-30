# linear-regression-practice

import numpy as np
import pandas as pd

df = pd.read_csv('Lemonade.csv')
df[:3]
df['Day'] = df['Day'].map({'Sunday':7, 
                           'Saturday':6, 
                           'Friday':5, 
                           'Thursday': 4, 
                           'Wednesday':3, 
                           'Tuesday':2, 
                           'Monday':1})

# 設定 X(features), y(label)
X, y = df[['Day', 'Temperature', 'Rainfall', 'Flyers', 'Price']].values, df[['Sales']].values
# 分成訓練、驗證資料(選擇性)

# 選擇演算法 餵入資料 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
# 預測 = Azure Score
model.predict(X)
# 評估 = Azure evaluate
print(model.score(X, y))
# 模型參數 y = Wx + b
model.coef_, model.intercept_

df['predict'] = model.predict(X)
df['residual'] = (df['Sales'] - df['predict']) /df['Sales'] * 100
df[:]

# 繪圖 殘差 = 真正與預測的差異 %
import matplotlib.pyplot as plt
plt.hist(df.residual, bins=12, edgecolor='red')
plt.show()
