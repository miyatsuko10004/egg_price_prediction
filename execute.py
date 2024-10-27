import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import openpyxl
from openpyxl.chart import LineChart, Reference

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

# SARIMAモデルの適用
# ここでは例としてSARIMA(1,1,1)(1,1,1,12)を使用していますが、
# 実際のデータに合わせて調整が必要かもしれません
sarima_model = SARIMAX(data['egg_price'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit()

# 2025年1月から2027年3月までの予測
forecast_index = pd.date_range(start='2025-01-01', end='2027-03-31', freq='MS')
sarima_forecast = sarima_fit.get_forecast(steps=len(forecast_index))
sarima_mean = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

print("2025年1月から2027年3月までの鶏卵価格予測 (SARIMA):")
print(sarima_mean)

# モデルの評価指標
mse = mean_squared_error(data['egg_price'][-12:], sarima_fit.fittedvalues[-12:])
rmse = np.sqrt(mse)
mae = mean_absolute_error(data['egg_price'][-12:], sarima_fit.fittedvalues[-12:])
r2 = r2_score(data['egg_price'][-12:], sarima_fit.fittedvalues[-12:])

print(f"\nモデル評価指標:")
print(f"平均二乗誤差 (MSE): {mse:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
print(f"平均絶対誤差 (MAE): {mae:.2f}")
print(f"決定係数 (R-squared): {r2:.2f}")

# 結果をDataFrameにまとめる
results = pd.DataFrame({
    'Date': forecast_index.strftime('%Y-%m'),  # 日付を'YYYY-MM'形式に変換
    'SARIMA_Forecast': sarima_mean.values,
    'Lower_CI': sarima_ci['lower egg_price'],
    'Upper_CI': sarima_ci['upper egg_price']
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

# Excelファイルを作成
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "予測結果"

# タイトルを追加
ws['A1'] = "2025年1月から2027年3月までの鶏卵価格予測結果 (SARIMA)"
ws['A1'].font = openpyxl.styles.Font(bold=True, size=14)

# 予測結果を追加
ws.append(['Date', 'SARIMA_Forecast', 'Lower_CI', 'Upper_CI'])  # ヘッダーを手動で追加
for r in results.itertuples(index=False, name=None):
    ws.append(r)

# 評価指標を追加
ws.append([])
ws.append(["モデル評価指標と説明"])
for r in metrics_df.itertuples(index=False, name=None):
    ws.append(r)

# グラフを作成
chart = LineChart()
chart.title = "鶏卵価格予測 (SARIMA)"
chart.x_axis.title = "日付"
chart.y_axis.title = "価格"

data = Reference(ws, min_col=2, min_row=1, max_col=4, max_row=len(results)+1)
cats = Reference(ws, min_col=1, min_row=2, max_row=len(results)+1)

chart.add_data(data, titles_from_data=True)
chart.set_categories(cats)

# グラフをシートに追加
ws.add_chart(chart, "G2")

# Excelファイルを保存
wb.save("result_sarima.xlsx")

print("\n結果がresult_sarima.xlsxファイルに出力されました。")
