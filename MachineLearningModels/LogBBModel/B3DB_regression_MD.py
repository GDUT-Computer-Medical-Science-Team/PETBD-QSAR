import pandas as pd
import numpy as np
import os
import joblib
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from utils.DataLogger import DataLogger
from datetime import datetime
import matplotlib.pyplot as plt
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/b3db_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "ratio_xgboost_training")

# [Chinese text removed]
b3db_file = "../../data/B3DB_regression.tsv"
desc_file = '../../data/B3DB_w_desc_regrssion_fp.csv'
desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"



# [Chinese text removed]
df = pd.read_csv(b3db_file, sep='\t')
RFE_features_to_select = 50
# [Chinese text removed]SMILES[Chinese text removed]
smile_column_name = 'SMILES'
pred_column_name = 'logBB'
seed = int(time())

# [Chinese text removed]
if not os.path.exists(desc_file):
    X = calculate_Mordred_desc(df[smile_column_name])  # [Chinese text removed]calculate_Mordred_desc[Chinese text removed]
    pd.concat([X, df[pred_column_name]], axis=1).to_csv(desc_file, index=False)
else:
    df = pd.read_csv(desc_file)
    X = df.drop([pred_column_name], axis=1)
    X = X.drop(smile_column_name, axis=1)

y = df[pred_column_name]
mean_value = df['logBB'].mean()  # [Chinese text removed]
df['logBB'].fillna(mean_value, inplace=True)  # [Chinese text removed] NaN [Chinese text removed]

# Feature normalization
sc = MinMaxScaler()
sc.fit(X)
X = pd.DataFrame(sc.transform(X))

# features[Chinese text removed]
if not os.path.exists(desc_index_file):
    log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
    log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
    # X = FeatureExtraction(X,
    #                       y,
    #                       VT_threshold=0.02,
    #                       RFE_features_to_select=RFE_features_to_select).feature_extraction()
    desc_index = (FeatureExtraction(X,
                                    y,
                                    VT_threshold=0.02,
                                    RFE_features_to_select=RFE_features_to_select).
                  feature_extraction(returnIndex=True, index_dtype=int))
    try:
        np.savetxt(desc_index_file, desc_index, fmt='%d')
        X = X[desc_index]
        log.info(f"features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}, [Chinese text removed]features[Chinese text removed]：{desc_index_file}")
    except (TypeError, KeyError) as e:
        log.error(e)
        os.remove(logBB_desc_index_file)
        sys.exit()
else:
    log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
    desc_index = np.loadtxt(desc_index_file, dtype=int, delimiter=',').tolist()
    X = X[desc_index]
    log.info(f"[Chinese text removed]features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}")

# [Chinese text removed]
# [Chinese text removed]，[Chinese text removed]KFold[Chinese text removed]Error
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# [Chinese text removed]Starting，[Chinese text removed]
model = joblib.load('./logbbModel/xgb_mordred_model.joblib')
# [Chinese text removed] X_val [Chinese text removed] y_val [Chinese text removed]
# [Chinese text removed]，[Chinese text removed]


# [Chinese text removed]
y_pred = model.predict(X)

# [Chinese text removed]
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# [Chinese text removed]MAPE，avoid division by zeroError
epsilon = 1e-10
mape = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), epsilon))) * 100

# [Chinese text removed] R [Chinese text removed]
n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# [Chinese text removed]
os.makedirs('./result', exist_ok=True)

# [Chinese text removed]originalB3DB[Chinese text removed]
original_b3db = pd.read_csv(b3db_file, sep='\t')

# [Chinese text removed]
if len(original_b3db) == len(y_pred):
    # [Chinese text removed]predicted values[Chinese text removed]original[Chinese text removed]
    original_b3db['Predicted_logBB'] = y_pred

    # [Chinese text removed]
    original_b3db.to_csv('./result/B3DB_logBB_Mordred_Predict.csv', index=False)
    log.info("[Chinese text removed] './result/B3DB_logBB_Mordred_Predict.csv'")
else:
    log.warning(f"original[Chinese text removed]({len(original_b3db)})[Chinese text removed]({len(y_pred)})[Chinese text removed]，[Chinese text removed]")

    # [Chinese text removed]SMILES、[Chinese text removed]predicted values[Chinese text removed]DataFrame
    y_pred_df = pd.DataFrame({
        'SMILES': df[smile_column_name],
        'Actual_logBB': y,
        'Predicted_logBB': y_pred
    })
    y_pred_df.to_csv('./result/B3DB_logBB_Mordred_Predict.csv', index=False)
    log.info("[Chinese text removed] './result/B3DB_logBB_Mordred_Predict.csv'")

# [Chinese text removed]
log.info("\n========[Chinese text removed] (B3DB[Chinese text removed])========")
log.info(f"samples[Chinese text removed]: {n}")
log.info(f"features[Chinese text removed]: {p}")
log.info(f"MSE: {mse:.4f}")
log.info(f"RMSE: {rmse:.4f}")
log.info(f"R²: {r2:.4f}")
log.info(f"[Chinese text removed]R²: {adj_r2:.4f}")
log.info(f"MAE: {mae:.4f}")
log.info(f"MAPE: {mape:.2f}%")

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.title('B3DB [Chinese text removed] - [Chinese text removed] vs predicted values')
plt.xlabel('[Chinese text removed] logBB [Chinese text removed]')
plt.ylabel('[Chinese text removed] logBB [Chinese text removed]')
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

# [Chinese text removed]
textstr = '\n'.join((
    f'R² = {r2:.3f}',
    f'RMSE = {rmse:.3f}',
    f'MAE = {mae:.3f}'
))
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('./result/B3DB_logBB_mordred_prediction_scatter.png')
plt.close()
