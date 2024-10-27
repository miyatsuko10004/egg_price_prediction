import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats

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

# 2025年1月から12月までの予測
forecast_index = pd.date_range(start='2025-01-01', end='2025-12-31', freq='MS')
es_forecast = es_fit.forecast(steps=len(forecast_index))
es_forecast.index = forecast_index

# 予測区間の計算（95%信頼区間）
residuals = es_fit.resid
std_resid = np.std(residuals)
conf_int = stats.norm.interval(0.95, loc=es_forecast, scale=std_resid)

print("2025年の鶏卵価格予測 (ExponentialSmoothing):")
print(es_forecast)

# モデルの評価指標
mse = mean_squared_error(data['egg_price'][-12:], es_fit.fittedvalues[-12:])
rmse = np.sqrt(mse)
mae = mean_absolute_error(data['egg_price'][-12:], es_fit.fittedvalues[-12:])
r2 = r2_score(data['egg_price'][-12:], es_fit.fittedvalues[-12:])

print(f"\nモデル評価指標:")
print(f"平均二乗誤差 (MSE): {mse:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
print(f"平均絶対誤差 (MAE): {mae:.2f}")
print(f"決定係数 (R-squared): {r2:.2f}")

# 結果をDataFrameにまとめる
results = pd.DataFrame({
    'Date': es_forecast.index,
    'ES_Forecast': es_forecast.values,
    'Lower_CI': conf_int[0],
    'Upper_CI': conf_int[1]
})

# 評価指標と説明を含む辞書
metrics_explanation = {
    'Metric': ['MSE', 'RMSE', 'MAE', 'R-squared', 'Confidence Interval'],
    'Value': [f'{mse:.2f}', f'{rmse:.2f}', f'{mae:.2f}', f'{r2:.2f}', '95%'],
    'Explanation': [
        '平均二乗誤差：予測誤差の二乗の平均。値が小さいほど良い。',
        '平方根平均二乗誤差：MSEの平方根。元のデータと同じ単位で誤差を表す。',
        '平均絶対誤差：予測誤差の絶対値の平均。',
        '決定係数：モデルの当てはまりの良さを0から1の間で表す。1に近いほど良い。',
        '予測値が95%の確率でこの区間内に収まると予想される範囲。'
    ]
}

# 評価指標のDataFrame
metrics_df = pd.DataFrame(metrics_explanation)

# 結果をCSVファイルとして出力
with open('result.csv', 'w', encoding='utf-8') as f:
    f.write("# 2025年の鶏卵価格予測結果\n\n")
    results.to_csv(f, index=False)
    f.write("\n# モデル評価指標と説明\n")
    metrics_df.to_csv(f, index=False)

print("\n結果がresult.csvファイルに出力されました。")
