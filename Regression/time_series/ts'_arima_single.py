import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.dates as mdates

# 換成自己的用戶行為軌跡資料
data_path = "~/matomo.csv"
df = pd.read_csv(data_path)

def create_feature_ts(col_name, sample_freq="D"):
    event_df = df[["tracking_time", col_name]][~df[col_name].isnull()]
    event_df["tracking_time"] = pd.to_datetime(event_df["tracking_time"])
    event_df.set_index("tracking_time", inplace=True)
    result_ts = event_df.groupby(col_name).resample(sample_freq).size()  # resample: 需注意 DataFrame 的索引為 Datetime
    result_ts = result_ts.unstack(fill_value=0)
    result_ts = result_ts.T
    return result_ts

# 分析 column_name_of_matomo 的時間序列
result_ts = create_feature_ts(column_name_of_matomo)

class ArimaModelerByColumn:
    def __init__(self, result_ts):
        '''
        result_ts: pd.DataFrame
        '''
        self.result_ts = result_ts

    def fit_predict(self, future_days=1, seasonal_days=7, freq="D", vertical_lines_weekday=None, plot_training_data=True, fig_path=None):
        '''
        parameters:

        future_days: int, 預測未來幾天
        seasonal_days: int, 週期性
        freq: str, 頻率
        vertical_lines_weekday: int, 畫出星期幾的垂直線
        plot_training_data: bool, 是否畫出訓練資料
        fig_path: str, 圖片存放路徑
        '''
        predictions = {}
        confidence_intervals = {}
        for col_name in self.result_ts.columns:
            print(f"Processing {col_name}...")

            auto_model = auto_arima(self.result_ts[col_name], seasonal=True, m=seasonal_days, trace=True, stepwise=True)
            print(auto_model.summary())
            model = ARIMA(self.result_ts[col_name], order=auto_model.order, seasonal_order=auto_model.seasonal_order)
            model_fit = model.fit()
            print(model_fit.summary())

            len_ts = len(self.result_ts[col_name])
            total_length = len_ts + future_days
            forecast_result = model_fit.get_forecast(steps=future_days)
            forecast_index = pd.date_range(start=self.result_ts.index[-1] + pd.Timedelta(days=1), periods=future_days, freq=freq)
            
            forecasted_values = pd.concat([model_fit.predict(start=1, end=len_ts, typ='levels'), forecast_result.predicted_mean])
            forecasted_values.index = pd.date_range(start=self.result_ts.index[0], periods=total_length, freq=freq)
            predictions[col_name] = forecasted_values
            
            ci = forecast_result.conf_int()
            ci.index = forecast_index
            confidence_intervals[col_name] = ci

            error = self.rmse(self.result_ts[col_name][:len_ts], forecasted_values[:len_ts])
            print(f"RMSE: {error}")

            self._plot_results(col_name, predictions[col_name], confidence_intervals[col_name], vertical_lines_weekday, plot_training_data, fig_path=fig_path)

        predictions_df = pd.DataFrame(predictions, index=pd.date_range(start=self.result_ts.index[0], periods=total_length, freq=freq))
        predictions_df.to_csv('arima_predictions.csv', index=True)
        print("Predictions saved to 'arima_predictions.csv'.")

    def _plot_results(self, col_name, predictions, confidence_interval, vertical_lines_weekday, plot_training_data, fig_path=None):
        '''
        parameters:
        
        col_name: str, 欄位名稱
        predictions: pd.Series, 預測值
        confidence_interval: pd.DataFrame, 信賴區間
        vertical_lines_weekday: int, 畫出星期幾的垂直線
        plot_training_data: bool, 是否畫出訓練資料
        fig_path: str, 圖片存放路徑，注意，這裡為資料夾不是 .png 路徑
        '''
        plt.figure(figsize=(10, 6))
        
        if plot_training_data:
            plt.plot(self.result_ts.index, self.result_ts[col_name], label='Original', color='blue')
            plt.plot(predictions.index[:len(self.result_ts)], predictions[:len(self.result_ts)], label='Fitted', linestyle='--', color='orange')
        
        plt.fill_between(confidence_interval.index, confidence_interval.iloc[:,0], confidence_interval.iloc[:,1], color='gray', alpha=0.2, label='95% Confidence Interval')
        plt.plot(predictions.index[len(self.result_ts):], predictions[len(self.result_ts):], label='Forecast', color='red')
        
        if vertical_lines_weekday is not None:
            self._draw_vertical_lines(vertical_lines_weekday, self.result_ts.index[0], predictions.index[-1])
        
        plt.title(f"{col_name} - Original vs Predicted")
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)) # 每週的星期一
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) # 月份和日
            
        plt.xticks(rotation=45) # 旋轉日期標籤以提高可讀性
        plt.tight_layout() # 自動調整子圖參數，使其填滿整個圖表區域

        if fig_path:
            fig_path = os.path.join(fig_path, f"{col_name}_arima.png")
            plt.savefig(fig_path)
        else:
            plt.show()

    def _draw_vertical_lines(self, weekday, start_date, end_date):
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=weekday))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == weekday:
                plt.axvline(x=current_date, color='gray', linestyle='--', alpha=0.7)
            current_date += pd.Timedelta(days=1)
        plt.text(current_date, ax.get_ylim()[1], weekdays[weekday], ha='center', va='bottom', rotation=45, alpha=0.7, fontsize=9)

    def rmse(self, x, y):
        return np.sqrt(np.mean((x-y)**2))



# 使用方式
if __name__ == "__main__":
    fig_path="~/single_arima/"
    arima = ArimaModelerByColumn(result_ts=result_ts)
    # 假設: 認為數據在每 seasonal 日有規律，想分析星期 N+1 的情況，建模後想預測未來 predict_day 天
    arima.fit_predict(future_days=predict_day, seasonal_days=seasonal, vertical_lines_weekday=N, plot_training_data=True, fig_path=fig_path)

