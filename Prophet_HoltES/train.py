import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import Holt
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # 忽略一些常见的警告信息

# --- 0. MAPE 计算函数 ---
def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE)。
    如果 y_true 中有0，则返回 np.nan，因为 MAPE 在这种情况下未定义或不稳定。
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if np.any(y_true == 0):
        print("  WARNING: Zero found in y_true for MAPE calculation. MAPE will be NaN.") # English Warning
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- 1. 数据加载与准备 ---

# 实际加载您的CSV文件
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("Successfully loaded train.csv and test.csv files.") # English message
except FileNotFoundError as e:
    print(f"ERROR: {e}. Please ensure 'train.csv' and 'test.csv' files exist in the script directory or provide the full path.") # English error
    exit()
except Exception as e:
    print(f"Error loading CSV files: {e}") # English error
    exit()

df_train['Time'] = pd.to_datetime(df_train['Time'])
df_train = df_train.set_index('Time')

df_test['Time'] = pd.to_datetime(df_test['Time'])
df_test = df_test.set_index('Time')

# 获取所有共同的Gxx列名 (以训练集为准，假设测试集包含这些列)
g_columns = [col for col in df_train.columns if col.startswith('G') and col in df_test.columns]
if not g_columns:
    print("ERROR: No common 'Gxx' columns found in the training and testing sets.") # English error
    exit()

# 存储每个模型在每个G列上的性能
results = {}

# --- 2. 针对每个Gxx列进行模型训练、预测与评估 ---
for col in g_columns:
    print(f"\n--- Processing Column: {col} ---") # English status

    # 数据提取与NaN值插补 (向前填充再向后填充)
    _train_series_orig = df_train[col]
    _test_series_orig = df_test[col]

    train_series = _train_series_orig.ffill().bfill()
    test_series = _test_series_orig.ffill().bfill() # 测试集也进行插补，用于评估

    # 诊断：检查插补后是否仍有NaN (主要针对整个序列都是NaN的极端情况)
    if train_series.isnull().all():
        print(f"  Column {col} train_series is all NaN after imputation. Skipping this column.") # English status
        results[col] = {"Error": "Training data is all NaN after imputation"}
        continue

    # 定义一个标志，判断测试集是否可用于评估 (即，并非全部为NaN)
    can_evaluate_metrics = not test_series.isnull().all()
    if not can_evaluate_metrics:
        print(
            f"  WARNING: Column {col} test_series (actual values for evaluation) is still all NaN after imputation. Models will attempt to predict, but metrics cannot be calculated, and actual test data will not be plotted.") # English warning

    if len(train_series) < 20:  # 训练数据太少
        print(f"  Column {col} has too few training data points ({len(train_series)}). Skipping processing.") # English status
        results[col] = {"Error": "Too few training data points"}
        continue

    if len(_test_series_orig) == 0: # 原始测试数据为空
        print(f"  Column {col} has no original test data points. Skipping processing.") # English status
        results[col] = {"Error": "Original test data is empty"}
        continue

    col_results = {}
    num_test_periods = len(df_test[col])  # 使用原始测试集的长度作为预测期数

    if num_test_periods == 0:
        print(f"  Column {col} has no data in df_test to determine prediction periods. Skipping.") # English status
        results[col] = {"Error": "Test dataset is empty, no prediction periods"}
        continue

    # 模型1: ARIMA
    print(f"  Training ARIMA model for {col}...") # English status
    try:
        arima_model = auto_arima(train_series,
                                 start_p=1, start_q=1,
                                 max_p=3, max_q=3,
                                 d=None, seasonal=False,
                                 trace=False, error_action='ignore',
                                 suppress_warnings=True, stepwise=True)
        arima_preds_values = arima_model.predict(n_periods=num_test_periods)
        arima_preds = pd.Series(arima_preds_values, index=df_test[col].index)

        if can_evaluate_metrics:
            arima_rmse = np.sqrt(mean_squared_error(test_series, arima_preds))
            arima_mae = mean_absolute_error(test_series, arima_preds)
            arima_mape = mean_absolute_percentage_error(test_series, arima_preds)
            col_results['ARIMA'] = {'RMSE': arima_rmse, 'MAE': arima_mae, 'MAPE': arima_mape, 'Predictions': arima_preds}
            print(f"  ARIMA for {col}: RMSE={arima_rmse:.6f}, MAE={arima_mae:.6f}, MAPE={arima_mape:.2f}%") # English status
        else:
            col_results['ARIMA'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'),
                                    'Predictions': arima_preds, 'Info': 'Test data all NaN'}
            print(f"  ARIMA for {col}: Predictions made, but test data is all NaN. Metrics cannot be calculated.") # English status
    except Exception as e:
        print(f"  ARIMA for {col} FAILED: {e}") # English error
        col_results['ARIMA'] = {'Error': str(e)}

    # 模型2: Prophet
    print(f"  Training Prophet model for {col}...") # English status
    try:
        prophet_train_df = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
        prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                                changepoint_prior_scale=0.01, growth='linear')
        prophet_model.fit(prophet_train_df)

        future_df = pd.DataFrame({'ds': df_test[col].index})
        prophet_forecast = prophet_model.predict(future_df)
        prophet_preds_values = prophet_forecast['yhat'].values
        prophet_preds = pd.Series(prophet_preds_values, index=df_test[col].index)

        if can_evaluate_metrics:
            prophet_rmse = np.sqrt(mean_squared_error(test_series, prophet_preds))
            prophet_mae = mean_absolute_error(test_series, prophet_preds)
            prophet_mape = mean_absolute_percentage_error(test_series, prophet_preds)
            col_results['Prophet'] = {'RMSE': prophet_rmse, 'MAE': prophet_mae, 'MAPE': prophet_mape, 'Predictions': prophet_preds}
            print(f"  Prophet for {col}: RMSE={prophet_rmse:.6f}, MAE={prophet_mae:.6f}, MAPE={prophet_mape:.2f}%") # English status
        else:
            col_results['Prophet'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'),
                                      'Predictions': prophet_preds, 'Info': 'Test data all NaN'}
            print(f"  Prophet for {col}: Predictions made, but test data is all NaN. Metrics cannot be calculated.") # English status
    except Exception as e:
        print(f"  Prophet for {col} FAILED: {e}") # English error
        col_results['Prophet'] = {'Error': str(e)}

    # 模型3: Holt's Exponential Smoothing
    print(f"  Training Holt's Exponential Smoothing model for {col}...") # English status
    try:
        holt_model = Holt(train_series, exponential=False, damped_trend=False).fit()
        holt_preds_values = holt_model.forecast(steps=num_test_periods)
        holt_preds = pd.Series(holt_preds_values, index=df_test[col].index)

        if can_evaluate_metrics:
            holt_rmse = np.sqrt(mean_squared_error(test_series, holt_preds))
            holt_mae = mean_absolute_error(test_series, holt_preds)
            holt_mape = mean_absolute_percentage_error(test_series, holt_preds)
            col_results['Holt'] = {'RMSE': holt_rmse, 'MAE': holt_mae, 'MAPE': holt_mape, 'Predictions': holt_preds}
            print(f"  Holt's ES for {col}: RMSE={holt_rmse:.6f}, MAE={holt_mae:.6f}, MAPE={holt_mape:.2f}%") # English status
        else:
            col_results['Holt'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'),
                                   'Predictions': holt_preds, 'Info': 'Test data all NaN'}
            print(f"  Holt's ES for {col}: Predictions made, but test data is all NaN. Metrics cannot be calculated.") # English status
    except Exception as e:
        print(f"  Holt's ES for {col} FAILED: {e}") # English error
        col_results['Holt'] = {'Error': str(e)}

    results[col] = col_results

    # --- 为当前列绘制所有模型的预测图 ---
    plt.figure(figsize=(15, 8))
    plt.plot(train_series.index, train_series, label='Train Data', color='blue', alpha=0.6)

    plot_title_parts = [f'Forecasts for Column: {col}'] # English Title
    actual_test_data_plotted = False

    if can_evaluate_metrics:
        plt.plot(test_series.index, test_series, label='Test Data (Actual)', color='green', linewidth=2) # English Legend
        actual_test_data_plotted = True
    else:
        plot_title_parts.append('[WARNING: Test actuals are all NaN, not plotted]') # English Warning

    # 绘制 ARIMA 预测
    arima_info = col_results.get('ARIMA', {})
    if isinstance(arima_info, dict) and 'Predictions' in arima_info:
        label_arima = 'ARIMA Predictions' # English Legend
        if can_evaluate_metrics:
            label_arima += f" (RMSE: {arima_info.get('RMSE', float('nan')):.4f}, MAE: {arima_info.get('MAE', float('nan')):.4f}, MAPE: {arima_info.get('MAPE', float('nan')):.2f}%)"
        plt.plot(arima_info['Predictions'].index, arima_info['Predictions'], label=label_arima, linestyle='--', alpha=0.8)
    elif isinstance(arima_info, dict) and 'Error' in arima_info:
        plot_title_parts.append(f"[ARIMA Error: {arima_info['Error']}]")

    # 绘制 Prophet 预测
    prophet_info = col_results.get('Prophet', {})
    if isinstance(prophet_info, dict) and 'Predictions' in prophet_info:
        label_prophet = 'Prophet Predictions' # English Legend
        if can_evaluate_metrics:
            label_prophet += f" (RMSE: {prophet_info.get('RMSE', float('nan')):.4f}, MAE: {prophet_info.get('MAE', float('nan')):.4f}, MAPE: {prophet_info.get('MAPE', float('nan')):.2f}%)"
        plt.plot(prophet_info['Predictions'].index, prophet_info['Predictions'], label=label_prophet, linestyle='-.', alpha=0.8)
    elif isinstance(prophet_info, dict) and 'Error' in prophet_info:
        plot_title_parts.append(f"[Prophet Error: {prophet_info['Error']}]")

    # 绘制 Holt's ES 预测
    holt_info = col_results.get('Holt', {})
    if isinstance(holt_info, dict) and 'Predictions' in holt_info:
        label_holt = "Holt's ES Predictions" # English Legend
        if can_evaluate_metrics:
            label_holt += f" (RMSE: {holt_info.get('RMSE', float('nan')):.4f}, MAE: {holt_info.get('MAE', float('nan')):.4f}, MAPE: {holt_info.get('MAPE', float('nan')):.2f}%)"
        plt.plot(holt_info['Predictions'].index, holt_info['Predictions'], label=label_holt, linestyle=':', alpha=0.8)
    elif isinstance(holt_info, dict) and 'Error' in holt_info:
        plot_title_parts.append(f"[Holt's ES Error: {holt_info['Error']}]")

    plt.title('\n'.join(plot_title_parts))
    plt.xlabel('Time') # English Label
    plt.ylabel('Value')   # English Label
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# --- 3. 结果汇总 ---
print("\n\n--- Prediction Performance Summary (RMSE) ---") # English Summary Title
summary_rmse_data = {}
for model_name in ['ARIMA', 'Prophet', 'Holt']:
    model_rmses = {}
    for col_name in g_columns:
        model_results = results.get(col_name, {}).get(model_name, {})
        model_rmses[col_name] = model_results.get('RMSE') if isinstance(model_results, dict) else None
    summary_rmse_data[model_name] = model_rmses
summary_rmse = pd.DataFrame(summary_rmse_data)
print(summary_rmse)

print("\n\n--- Prediction Performance Summary (MAE) ---") # English Summary Title
summary_mae_data = {}
for model_name in ['ARIMA', 'Prophet', 'Holt']:
    model_maes = {}
    for col_name in g_columns:
        model_results = results.get(col_name, {}).get(model_name, {})
        model_maes[col_name] = model_results.get('MAE') if isinstance(model_results, dict) else None
    summary_mae_data[model_name] = model_maes
summary_mae = pd.DataFrame(summary_mae_data)
print(summary_mae)

print("\n\n--- Prediction Performance Summary (MAPE) ---") # English Summary Title
summary_mape_data = {}
for model_name in ['ARIMA', 'Prophet', 'Holt']:
    model_mapes = {}
    for col_name in g_columns:
        model_results = results.get(col_name, {}).get(model_name, {})
        model_mapes[col_name] = model_results.get('MAPE') if isinstance(model_results, dict) else None
    summary_mape_data[model_name] = model_mapes
summary_mape = pd.DataFrame(summary_mape_data)
print(summary_mape)


print("\n--- Task Complete ---") # English Completion Message
