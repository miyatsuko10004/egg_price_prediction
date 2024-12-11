import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from scipy import stats
import openpyxl
from datetime import datetime, timedelta
import warnings

class RobustEggPriceForecast:
    def __init__(self, filepath):
        """
        外れ値を考慮した鶏卵価格予測モデル
        
        :param filepath: CSVファイルのパス
        """
        self.original_data = self.load_data(filepath)
        self.processed_data = self.preprocess_data()
    
    def load_data(self, filepath):
        """
        データの読み込み
        
        :param filepath: CSVファイルのパス
        :return: 読み込んだデータフレーム
        """
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'] + '/1', format='%Y/%m/%d')
        data.set_index('Date', inplace=True)
        
        # 数値列の変換
        numeric_columns = [
            'egg_price', 'egg_production', 'egg_shipment', 'egg_recipt', 
            'chick_count', 'farmer_price', 'wholesale_price', 'retail_price', 
            'household_consumption_per_man', 'egg_import', 
            'chick_feed_shipment', 'chicken_feed_shipment', 'feed_price'
        ]
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data.dropna()
    
    def preprocess_data(self):
        """
        データの前処理と特徴量エンジニアリング
        
        :return: 前処理されたデータフレーム
        """
        data = self.original_data.copy()
        
        # 移動平均と標準偏差の追加
        data['price_ma_3m'] = data['egg_price'].rolling(window=3).mean()
        data['price_ma_6m'] = data['egg_price'].rolling(window=6).mean()
        data['price_std_3m'] = data['egg_price'].rolling(window=3).std()
        
        return data.dropna()
    
    def detect_outliers(self, method='iqr'):
        """
        外れ値の検出
        
        :param method: 外れ値検出方法 ('iqr' or 'zscore')
        :return: 外れ値のインデックス
        """
        if method == 'iqr':
            # 四分位範囲法
            Q1 = self.processed_data['egg_price'].quantile(0.25)
            Q3 = self.processed_data['egg_price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.processed_data[
                (self.processed_data['egg_price'] < lower_bound) | 
                (self.processed_data['egg_price'] > upper_bound)
            ]
        
        elif method == 'zscore':
            # Zスコア法
            z_scores = np.abs(stats.zscore(self.processed_data['egg_price']))
            outliers = self.processed_data[z_scores > 3]
        
        return outliers
    
    def robust_preprocessing(self):
        """
        外れ値に対してロバストな前処理
        
        :return: 外れ値処理されたデータフレーム
        """
        # 外れ値の検出（2つの方法）
        iqr_outliers = self.detect_outliers(method='iqr')
        zscore_outliers = self.detect_outliers(method='zscore')
        
        # データのコピー
        data_processed = self.processed_data.copy()
        
        # 外れ値の処理（中央値補完）
        combined_outliers = pd.concat([iqr_outliers, zscore_outliers]).drop_duplicates()
        
        for col in ['egg_price', 'price_ma_3m', 'price_ma_6m', 'price_std_3m']:
            median_value = data_processed[col].median()
            data_processed.loc[combined_outliers.index, col] = median_value
        
        return data_processed, combined_outliers
    
    def create_sarima_model(self, data):
        """
        ロバストSARIMAモデルの作成
        
        :param data: モデル学習用データ
        :return: 学習済みSARIMAモデル
        """
        # 外れ値の影響を考慮した特徴量
        features = [
            'price_ma_3m', 
            'price_ma_6m', 
            'price_std_3m'
        ]
        
        exog = data[features]
        
        sarima_model = SARIMAX(
            data['egg_price'], 
            exog=exog,
            order=(1,1,1), 
            seasonal_order=(1,1,1,12)
        )
        
        return sarima_model.fit()
    
    def forecast_with_uncertainty(self, sarima_fit, forecast_steps):
        """
        不確実性を考慮した予測
        
        :param sarima_fit: 学習済みSARIMAモデル
        :param forecast_steps: 予測ステップ数
        :return: 予測結果
        """
        # 予測期間の設定
        forecast_index = pd.date_range(
            start=self.processed_data.index[-1] + pd.offsets.MonthBegin(1), 
            periods=forecast_steps, 
            freq='MS'
        )
        
        # 予測のための外生変数の準備
        last_data_point = self.processed_data.iloc[-1]
        exog_forecast = pd.DataFrame({
            'price_ma_3m': [last_data_point['price_ma_3m']] * len(forecast_index),
            'price_ma_6m': [last_data_point['price_ma_6m']] * len(forecast_index),
            'price_std_3m': [last_data_point['price_std_3m']] * len(forecast_index)
        }, index=forecast_index)
        
        # 予測の実行
        forecast = sarima_fit.get_forecast(
            steps=len(forecast_index), 
            exog=exog_forecast
        )
        
        return {
            'forecast_mean': forecast.predicted_mean,
            'forecast_conf_int': forecast.conf_int()
        }
    
    def generate_report(self, forecast_results, outliers):
        """
        予測結果とレポートの生成
        
        :param forecast_results: 予測結果
        :param outliers: 検出された外れ値
        """
        # Excelワークブックの作成
        wb = openpyxl.Workbook()
        
        # 予測結果シート
        ws_forecast = wb.active
        ws_forecast.title = "Price_Forecast"
        
        # ヘッダーの書き込み
        headers = ['Date', 'Forecast_Mean', 'Lower_CI', 'Upper_CI']
        ws_forecast.append(headers)
        
        # 予測結果の書き込み
        for date in forecast_dates:
            # インデックスが一致するように調整
            lower_ci = forecast_conf_int.loc[date, 'lower egg_price']
            upper_ci = forecast_conf_int.loc[date, 'upper egg_price']
            
            ws_forecast.append([
                date.strftime('%Y/%m'), 
                forecast_value, 
                lower_ci, 
                upper_ci
            ])
        
        # 外れ値シートの作成
        ws_outliers = wb.create_sheet(title="Outliers")
        ws_outliers.append(['Date', 'Egg_Price', 'Detected_Method'])
        
        for idx, row in outliers.iterrows():
            ws_outliers.append([
                idx.strftime('%Y/%m'), 
                row['egg_price'], 
                '合成外れ値検出法'
            ])
        
        # レポートの保存
        wb.save('robust_egg_price_forecast.xlsx')
    
    def run_forecast(self):
        """
        価格予測の実行
        
        :return: 予測結果
        """
        # ロバスト前処理
        data_processed, outliers = self.robust_preprocessing()
        
        # モデルの学習
        sarima_fit = self.create_sarima_model(data_processed)
        
        # 予測の実行
        forecast_results = self.forecast_with_uncertainty(
            sarima_fit, 
            forecast_steps=24  # 2年分
        )
        
        # レポート生成
        self.generate_report(forecast_results, outliers)
        
        return forecast_results

def main():
    # モデルの初期化と予測実行
    model = RobustEggPriceForecast("eggData.csv")
    forecast_results = model.run_forecast()
    
    # 予測結果の表示
    print("鶏卵価格予測結果:")
    print(forecast_results['forecast_mean'])

if __name__ == "__main__":
    main()