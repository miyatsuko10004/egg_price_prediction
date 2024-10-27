import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# データの読み込み
data = pd.read_csv("eggData.csv")

# 必要なデータの整形
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


model = ExponentialSmoothing(data['egg_price'], trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit()
forecast = fit.forecast(steps=12)  # 1年先まで予測

X = data[['egg_production', 'egg_shipment', 'egg_recipt',	'chick_count',	'farmer_price',	'wholesale_price',	'retail_price',	'household_consumption_per_man',	'egg_import',	'chick_feed_shipment', 'chicken_feed_shipment',	'feed_price']]
y = data['Egg_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ランダムフォレストによる学習と予測
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 評価
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
