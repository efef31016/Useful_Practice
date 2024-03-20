# 計算
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 檢定與繪圖
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 回歸模型
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error



class GrangerCausality:
    def __init__(self, data, max_lag=4, significance_level=0.05):
        '''
        data - index: Timestamp, columns: vary ts (features)
        '''
        self.data = data
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.significant_relations = {}
        self.x_y_fit_data = {}

    def find_granger_relations(self):
        '''
        parameters:
        result_ts: DataFrame, index: timestamp, column: ts_i

        output:
        {A: [(B1, B1_pvalue, B1_log), (B2, B2_pvalue, B2_log), ..., (Bn, Bn_pvalue, Bn_log)]}, where Bi Granger-causes A
        '''
        for i in self.data.columns:
            for j in self.data.columns:
                if i != j:
                    test_result = grangercausalitytests(self.data[[i, j]], self.max_lag, verbose=False)
                    p_values = [test_result[lag + 1][0]['ssr_chi2test'][1] for lag in range(self.max_lag)]
                    min_p_value = min(p_values)

                    if min_p_value < self.significance_level:
                        min_p_lag = p_values.index(min_p_value) + 1

                        if i not in self.significant_relations:
                            self.significant_relations[i] = [(j, min_p_value, min_p_lag)]
                        else:
                            self.significant_relations[i].append((j, min_p_value, min_p_lag))

        return self.significant_relations

    def prepare_fit_data(self):
        sig_granger = self.find_granger_relations()

        for key, value in sig_granger.items():
            max_log = max(value, key=lambda x: x[2])[2]

            x_train = pd.DataFrame()
            for granger_item in value:
                x_train_new = self.data[granger_item[0]].shift(granger_item[2])
                x_train = pd.concat([x_train, x_train_new], axis=1)

            x_train = x_train[max_log:].dropna()
            y_train = self.data[key][max_log:]

            self.x_y_fit_data[key] = (x_train.to_numpy(dtype=np.float64), y_train.to_numpy(dtype=np.float64).reshape(-1, 1))

        return self.x_y_fit_data
    


class GrangerGraph:
    def __init__(self, granger_data):
        '''
        granger_data: dict, output from granger_relation function
        '''
        self.granger_data = granger_data
        self.G = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        '''
        Internal method to build the graph from the Granger causality data
        '''
        for source, targets in self.granger_data.items():
            for target, p_value, lag in targets:
                # Edge thickness will be calculated during drawing based on p-value
                self.G.add_edge(source, target, p_value=p_value, lag=lag)

    def draw_graph(self):
        '''
        Draws the graph with line thickness determined by the significance level.
        Lines are made wider to ensure each is clearly visible.
        '''
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(self.G, seed=42)  # Node position layout

        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=5000, node_color='skyblue', alpha=0.6)

        # Draw edges with modified width calculation for better visibility
        edges = self.G.edges(data=True)
        # Increase the base of line width to make the lines wider and more distinct
        edge_widths = [1-10*data['p_value'] for u, v, data in edges]  
        nx.draw_networkx_edges(self.G, pos, arrowstyle='-|>', arrowsize=20, width=edge_widths)

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, font_size=12)

        plt.title('Granger Causality Graph')
        plt.axis('off')
        plt.show()



class RegresionModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def linear_regression(self):
        model_linear = LinearRegression()

        model_linear.fit(self.x_train, self.y_train)

        y_pred_linear = model_linear.predict(self.x_train)

        mse_linear = mean_squared_error(self.y_train, y_pred_linear)
        # print(f"Linear Regression MSE: {mse_linear}")
        
        return model_linear, mse_linear

    def ridge_regression(self, alpha=1.0):
        model_ridge = Ridge(alpha=alpha)

        model_ridge.fit(self.x_train, self.y_train)

        y_pred_ridge = model_ridge.predict(self.x_train)

        mse_ridge = mean_squared_error(self.y_train, y_pred_ridge)
        # print(f"Ridge Regression MSE: {mse_ridge}")

        return model_ridge, mse_ridge

    def random_forest_regression(self, n_estimators=100):
        model_rf = RandomForestRegressor(n_estimators=n_estimators)

        model_rf.fit(self.x_train, self.y_train)

        y_pred_rf = model_rf.predict(self.x_train)

        mse_rf = mean_squared_error(self.y_train, y_pred_rf)
        # print(f"Random Forest Regression MSE: {mse_rf}")

        return model_rf, mse_rf

    def grad_boos_regression(self, n_estimators=100, learning_rate=0.1):
        model_gbr = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

        model_gbr.fit(self.x_train, self.y_train)

        y_pred_gbr = model_gbr.predict(self.x_train)

        mse_gbr = mean_squared_error(self.y_train, y_pred_gbr)
        # print(f"Gradient Boosting Regression MSE: {mse_gbr}")

        return model_gbr, mse_gbr

    def svr_regression(self, kernel="rbf"):
        model_svr = SVR(kernel=kernel)

        model_svr.fit(self.x_train, self.y_train)

        y_pred_svr = model_svr.predict(self.x_train)

        mse_svr = mean_squared_error(self.y_train, y_pred_svr)
        # print(f"Support Vector Regression MSE: {mse_svr}")

        return model_svr, mse_svr

    def lasso_regression(self, alpha=1.0):
        model_lasso = Lasso(alpha=alpha)

        model_lasso.fit(self.x_train, self.y_train)

        y_pred_lasso = model_lasso.predict(self.x_train)

        mse_lasso = mean_squared_error(self.y_train, y_pred_lasso)
        # print(f"Lasso Regression MSE: {mse_lasso}")

        return model_lasso, mse_lasso

    def elastic_regression(self, alpha=1.0, l1_ratio=0.5):
        model_elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        model_elastic.fit(self.x_train, self.y_train)

        y_pred_elastic = model_elastic.predict(self.x_train)

        mse_elastic = mean_squared_error(self.y_train, y_pred_elastic)
        # print(f"Elastic Net Regression MSE: {mse_elastic}")

        return model_elastic, mse_elastic



if __name__ == "__main__":

    # 匯入符合格式的多維度時間序列資料
    result_ts = pd.read_csv("multi-dim_ts.csv")

    # 建立該物件
    data = GrangerCausality(result_ts)

    # 可視化 Granger 因果關係
    granger_relation_data = data.find_granger_relations()
    graph = GrangerGraph(granger_relation_data)
    graph.draw_graph()

    # 回歸模型挑選
    x_y_fit_data = data.prepare_fit_data()
    for key, value in x_y_fit_data.items():
        print("="*35 + key + "="*35)
        regression_models = RegresionModel(value[0], value[1])
        li_model, li_error = regression_models.linear_regression()
        ri_mofrl, ri_error = regression_models.ridge_regression()
        rf_model, rf_error = regression_models.random_forest_regression()
        gb_model, gb_error = regression_models.grad_boos_regression()
        svr_model, svr_error = regression_models.svr_regression()
        lasso_model, lasso_error = regression_models.lasso_regression()
        elt_model, elt_error = regression_models.elastic_regression()
        
        all_model = [li_model, ri_mofrl, rf_model, gb_model, svr_model, lasso_model, elt_model]
        all_error = [li_error, ri_error, rf_error, gb_error, svr_error, lasso_error, elt_error]
        
        print(f"Best Model \'{all_model[all_error.index(min(all_error))]}\' with error \'{min(all_error)}\'")