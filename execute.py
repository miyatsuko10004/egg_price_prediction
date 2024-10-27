import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# データの読み込み
data = pd.read_csv("eggData.csv")

# 'Date'列を適切な日付形式に変換
data['Date'] = pd.to_datetime(data['Date'] + '/1', format='%Y/%m/%d')
data.set_index('Date', inplace=True)

# データ型の確認と変換
numeric_columns = ['egg_price', 'egg_production', 'egg_shipment', 'egg_recipt', 'chick_count', 'farmer_price', 'wholesale_price', 'retail_price', 'household_consumption_per_man', 'egg_import', 'chick_feed_shipment', 'chicken_feed_shipment', 'feed_price']

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 欠損値の処理
data = data.dropna()

# データ型の確認
print(data.dtypes)
print(data['egg_price'].head())

# ExponentialSmoothingモデルの適用
es_model = ExponentialSmoothing(data['egg_price'].astype(float), trend='add', seasonal='add', seasonal_periods=12)
es_fit = es_model.fit()
es_forecast = es_fit.forecast(steps=12)  # 1年先まで予測

# ランダムフォレストモデルの準備
X = data[['egg_production', 'egg_shipment', 'egg_recipt', 'chick_count', 'farmer_price', 'wholesale_price', 'retail_price', 'household_consumption_per_man', 'egg_import', 'chick_feed_shipment', 'chicken_feed_shipment', 'feed_price']]
y = data['egg_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ランダムフォレストによる学習と予測
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# 評価
mae = mean_absolute_error(y_test, rf_predictions)
print(f"平均絶対誤差: {mae}")

# 結果をDataFrameにまとめる
results = pd.DataFrame({
    'Date': y_test.index,
    'Actual_Price': y_test.values,
    'RF_Predicted_Price': rf_predictions,
    'ES_Forecast': [es_forecast[0]] * len(y_test)  # ExponentialSmoothingの予測値（簡略化のため同じ値を使用）
})

# 結果をCSVファイルとして出力
results.to_csv('result.csv', index=False)

print("結果がresult.csvファイルに出力されまし。")
