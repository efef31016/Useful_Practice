import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.dates as mdates
from joblib import Parallel, delayed

from datetime import datetime, timedelta
import json
import os

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
    def __init__(self, result_ts, future_days=1, seasonal_days=7, freq="D", fig_path=None):
        '''
        parameters:

        result_ts: DataFrame; 以column為key的時間序列資料
        future_days: int; 欲預測的天數
        seasonal_days: int; 猜測的季節性天數
        freq: str; 以何種頻率進行時間序列分析
        fig_path: str; 儲存預測結果的路徑
        '''
        self.result_ts = result_ts
        self.future_days = future_days
        self.seasonal_days = seasonal_days
        self.freq = freq
        self.fig_path = fig_path
        self.steep_cols = set()

    def log_msg(msg, log_file="process.log"):
        '''
        儲存日誌，將刪除三天前的日誌
        '''
        current_time = datetime.now()
        three_days_ago = current_time - timedelta(days=3)
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

        new_logs = []

        # 去掉三天前的日誌
        try:
            with open(log_file, 'r') as file:
                for line in file:
                    try:
                        timestamp_str, _ = line.split(' - ', 1)
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        if timestamp > three_days_ago:
                            new_logs.append(line)
                    except ValueError:
                        # 解析失敗，忽略
                        pass
        except FileNotFoundError:
            # 若文件不存在，忽略
            pass

        # 添加新的日誌
        new_logs.append(f"{current_time_str} - {msg}\n")

        # 重新寫入日誌文件
        with open(log_file, 'w') as file:
            file.writelines(new_logs)

    def rmse(self, x, y):
        return np.sqrt(np.mean((x-y)**2))
    
    def process_column(self, col_name):
        '''
        透過 AIC 選擇 ARIMA 模型，並進行預測
        '''
        print(f"Processing {col_name}...")
        try:
            auto_model = auto_arima(self.result_ts[col_name], seasonal=True, m=self.seasonal_days, trace=True, stepwise=True)
            print(auto_model.summary())

            model = ARIMA(self.result_ts[col_name], order=auto_model.order, seasonal_order=auto_model.seasonal_order)
            model_fit = model.fit()

            len_ts = len(self.result_ts[col_name])
            total_length = len_ts + self.future_days
            forecast_result = model_fit.get_forecast(steps=self.future_days)
            forecast_index = pd.date_range(start=self.result_ts.index[-1] + pd.Timedelta(days=1), periods=self.future_days, freq=self.freq)
            
            forecasted_values = pd.concat([model_fit.predict(start=1, end=len_ts, typ='levels'), forecast_result.predicted_mean])
            forecasted_values.index = pd.date_range(start=self.result_ts.index[0], periods=total_length, freq=self.freq) 
            
            ci = forecast_result.conf_int()
            ci.index = forecast_index

            error = self.rmse(self.result_ts[col_name][:len_ts], forecasted_values[:len_ts])

        except Exception as e:
            self.log_msg(f"Error processing {col_name}: {e}\n")

            self.steep_cols.add(col_name)
            model_fit = None
            forecasted_values = None
            ci = None
            error = -1

        return col_name, error, model_fit, forecasted_values, ci

    def parallel_fit_predict(self):
        '''
        並行透過 AIC 選擇每個 column 的 ARIMA 模型，並進行預測
        '''
        results = Parallel(n_jobs=-1)(
            delayed(self.process_column)(col_name) for col_name in self.result_ts.columns
        )
        return results
    
    def save_predictions(self, predictions_df):
        '''
        儲存預測結果
        '''
        self.log_msg("Predictions saved to 'arima_predictions.csv'.")
        predictions_df.to_csv(os.path.join(self.fig_path, "arima_predictions.csv"), index=True)

    def plot_results(self, col_name, predictions, confidence_interval, vertical_lines_weekday, plot_training_data, fig_path=None):
        '''
        繪製預測結果
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
        '''
        繪製垂直線，表示每週的星期幾
        '''
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



# 使用範例
if __name__ == "__main__":

    ################################################################################################
    # 根據不同的seasonal，選出最佳模型並保存對應結果，其中包含以下五點
    # 1. 最小 training error (RMSE)
    # 2. 季節
    # 3. 模型
    # 4. 預測結果
    # 5. 預測未來的信賴區間
    fig_path = "~/single_arima"
    season_guess = [i for i in range(1, 31)]
    best_seasonal = {}
    unstationary_cols = set()

    print("choose best seasonal_day:...")
    for seasonal_day in season_guess:
        start = datetime.now()
        arima = ArimaModelerByColumn(result_ts=result_ts, future_days=7, seasonal_days=seasonal_day, freq="D", fig_path=fig_path)
        results = arima.parallel_fit_predict()

        for col_name, error, model_fit, forcaste_value, ci in results:
            if error == 0:
                unstationary_cols.add(col_name)
                
            if col_name not in best_seasonal or error < best_seasonal[col_name][0]:
                best_seasonal[col_name] = (error, seasonal_day, model_fit, forcaste_value, ci)
        end = datetime.now()
        print("seasonal_day {} spends {:.2f} seconds".format(seasonal_day, (end-start).total_seconds()))

    ################################################################################################
    # 畫圖及儲存預測結果與最佳季節
    seasonal = {}
    predict_dict = {}
    for col_name, value in best_seasonal.items():
        '''
        value: (error, seasonal_day, model_fit, forcaste_value, ci)
        '''
        if value[2]:
            seasonal[col_name] = value[1]
            arima.plot_results(col_name, predictions=value[3], confidence_interval=value[4], vertical_lines_weekday=2, plot_training_data=True, fig_path=fig_path)
            predict_dict[col_name] = value[3]

    arima.save_predictions(pd.DataFrame(predict_dict))

    with open(os.path.join(fig_path, "best_seasonal.json"), 'w') as file:
        json.dump(seasonal, file)