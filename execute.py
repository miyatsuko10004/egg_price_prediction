import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from itertools import combinations

# データの読み込み
data = pd.read_csv("eggData.csv")

# 'Date'列を適切な日付形式に変換
data['Date'] = pd.to_datetime(data['Date'] + '/1', format='%Y/%m/%d')
data.set_index('Date', inplace=True)

# データ型の確認と変換
numeric_columns = ['egg_price', 'egg_production', 'egg_shipment', 'egg_recipt', 'chick_count', 
                   'farmer_price', 'wholesale_price', 'retail_price', 'household_consumption_per_man', 
                   'egg_import', 'chick_feed_shipment', 'chicken_feed_shipment', 'feed_price']

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 欠損値の処理
data = data.dropna()

# 1. 相関分析
correlation_matrix = data.corr()
correlation_with_price = correlation_matrix['egg_price'].sort_values(ascending=False)
print("鶏卵価格との相関係数:")
print(correlation_with_price)

# 2. ランダムフォレストによる特徴量重要度の評価
X = data[numeric_columns].drop(columns=['egg_price'])
y = data['egg_price']

# データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデル
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# 特徴量重要度を取得
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\n特徴量重要度 (ランダムフォレスト):")
print(feature_importances.sort_values(ascending=False))

# 3. SARIMAXモデルのグリッドサーチで最適変数を特定
best_aic = np.inf
best_model = None
best_features = None

# 説明変数の組み合わせをすべて試す
for k in range(1, len(numeric_columns)):  # 説明変数の数を1から試す
    for combo in combinations(numeric_columns[1:], k):  # egg_priceを除く
        try:
            sarimax_model = SARIMAX(data['egg_price'], 
                                    exog=data[list(combo)], 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 1, 12))
            sarimax_fit = sarimax_model.fit(disp=False)
            if sarimax_fit.aic < best_aic:
                best_aic = sarimax_fit.aic
                best_model = sarimax_fit
                best_features = combo
        except Exception as e:
            continue

print(f"\n最適モデルのAIC: {best_aic}")
print(f"選択された説明変数: {best_features}")

# 4. 最適モデルの評価
if best_model:
    steps = 12
    forecast = best_model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    ci = forecast.conf_int()

    print("\n次12か月の予測値:")
    print(forecast_mean)

    # 評価指標の計算
    mse = mean_squared_error(data['egg_price'][-12:], best_model.fittedvalues[-12:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data['egg_price'][-12:], best_model.fittedvalues[-12:])
    r2 = r2_score(data['egg_price'][-12:], best_model.fittedvalues[-12:])

    print(f"\n評価指標:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")