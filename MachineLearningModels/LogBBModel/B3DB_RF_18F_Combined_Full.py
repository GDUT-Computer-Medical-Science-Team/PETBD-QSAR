"""
B3DB External Validation Script using Random Forest with 18F Combined Features
[Chinese text removed]Random Forest 18F[Chinese text removed]features[Chinese text removed]B3DB[Chinese text removed]

Features used: Morgan Fingerprints + Mordred Descriptors + Metadata
Model: rf_combined_18F_model.joblib (trained with 18F resampling strategy)
Dataset: Full B3DB dataset (all samples)
"""

import pandas as pd
import numpy as np
import os
import joblib
import sys
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

sys.path.append('../../')
from utils.DataLogger import DataLogger
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc

log = DataLogger(log_file=f"../../data/b3db_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("b3db_rf_18F_full")

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """Calculate Morgan fingerprints for a list of SMILES"""
    fingerprints = []
    valid_smiles = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
            valid_smiles.append(smiles)
        else:
            log.warning(f"Invalid SMILES: {smiles}")

    return np.array(fingerprints), valid_smiles

def process_metadata_features(df):
    """
    Process metadata features for B3DB dataset
    Note: B3DB dataset doesn't have metadata, so we'll use default values
    """
    metadata_df = pd.DataFrame()

    # B3DB dataset typically doesn't have these columns, so we'll use default values
    # Default: mouse (1), male (1), median weight, median dosage
    n_samples = len(df)

    metadata_df['animal_type'] = [1] * n_samples  # Default: mouse
    metadata_df['gender'] = [1] * n_samples  # Default: male
    metadata_df['animal_weight'] = [25.0] * n_samples  # Default: 25g (typical mouse weight)
    metadata_df['injection_dosage'] = [100.0] * n_samples  # Default: 100 μCi

    log.info("Using default metadata values for B3DB dataset (no experimental metadata available)")

    return metadata_df

def calculate_combined_features(df, smiles_column='SMILES'):
    """
    Calculate combined features: Morgan + Mordred + Metadata
    """
    log.info("Calculating Mordred descriptors...")
    df_with_mordred = calculate_Mordred_desc(df[smiles_column])

    mordred_cols = [col for col in df_with_mordred.columns if col not in [smiles_column]]
    X_mordred = df_with_mordred[mordred_cols]

    valid_smiles = df_with_mordred[smiles_column].tolist()
    df_filtered = df[df[smiles_column].isin(valid_smiles)].reset_index(drop=True)
    X_mordred = X_mordred.reset_index(drop=True)

    log.info(f"After Mordred filtering: {len(df_filtered)} samples remain")

    log.info("Calculating Morgan fingerprints...")
    X_morgan, _ = calculate_morgan_fingerprints(df_filtered[smiles_column].tolist())

    log.info("Processing metadata features...")
    X_metadata = process_metadata_features(df_filtered)

    log.info(f"Dimension check:")
    log.info(f"  - Morgan fingerprints: {X_morgan.shape}")
    log.info(f"  - Mordred descriptors: {X_mordred.shape}")
    log.info(f"  - Metadata features: {X_metadata.shape}")

    if not (X_morgan.shape[0] == X_mordred.shape[0] == X_metadata.shape[0]):
        raise ValueError(f"Feature dimension mismatch: Morgan={X_morgan.shape[0]}, Mordred={X_mordred.shape[0]}, Metadata={X_metadata.shape[0]}")

    log.info("Combining all features...")
    X_combined = np.concatenate([X_morgan, X_mordred.values, X_metadata.values], axis=1)

    morgan_names = [f'Morgan_FP_{i}' for i in range(X_morgan.shape[1])]
    mordred_names = [f'Mordred_{col}' for col in X_mordred.columns]
    metadata_names = X_metadata.columns.tolist()

    all_feature_names = morgan_names + mordred_names + metadata_names

    log.info(f"Combined features shape: {X_combined.shape}")
    log.info(f"  - Morgan fingerprints: {X_morgan.shape[1]}")
    log.info(f"  - Mordred descriptors: {X_mordred.shape[1]}")
    log.info(f"  - Metadata features: {X_metadata.shape[1]}")

    return X_combined, all_feature_names, df_filtered

def main():
    log.info("="*80)
    log.info("Starting B3DB Full Dataset Validation with Random Forest 18F Combined Model")
    log.info("="*80)

    # File paths
    b3db_file = "../../data/B3DB_classification.tsv"
    model_file = './logbbModel/rf_fp_combined_18F_model.joblib'
    scaler_file = './logbbModel/xgb_combined_18F_scaler.joblib'  # Using XGB scaler (RF may not have separate scaler)
    feature_index_file = '../../data/logBB_data/RF_FP_18F_feature_index.txt'

    # Check if model files exist
    if not os.path.exists(model_file):
        log.error(f"Model file not found: {model_file}")
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if not os.path.exists(scaler_file):
        log.error(f"Scaler file not found: {scaler_file}")
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

    if not os.path.exists(feature_index_file):
        log.error(f"Feature index file not found: {feature_index_file}")
        raise FileNotFoundError(f"Feature index file not found: {feature_index_file}")

    # Load B3DB dataset
    log.info("Loading B3DB dataset...")
    df = pd.read_csv(b3db_file, sep='\t')
    log.info(f"Original B3DB data shape: {df.shape}")

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'

    # Check for samples with valid logBB values
    df_with_logbb = df[df[pred_column_name].notna()].copy()
    n_with_logbb = len(df_with_logbb)
    log.info(f"Samples with valid logBB values: {n_with_logbb}")
    log.info(f"Samples without logBB values: {len(df) - n_with_logbb}")

    # Calculate combined features for ALL samples (not filtering by logBB)
    log.info("Calculating combined features (Morgan + Mordred + Metadata) for full dataset...")
    X_combined, all_feature_names, df_filtered = calculate_combined_features(df, smiles_column=smile_column_name)

    log.info(f"Feature matrix shape: {X_combined.shape}")

    # Handle missing values and infinities
    log.info("Handling missing values and infinities...")
    X_combined = np.where(np.isinf(X_combined), np.nan, X_combined)

    imputer = SimpleImputer(strategy='mean')
    X_combined = imputer.fit_transform(X_combined)

    # Load feature indices
    log.info("Loading feature indices...")
    with open(feature_index_file, 'r') as f:
        selected_indices = [int(line.strip()) for line in f.readlines()]

    log.info(f"Number of selected features: {len(selected_indices)}")

    # Select features
    max_idx = max(selected_indices) if selected_indices else 0
    if max_idx >= X_combined.shape[1]:
        log.error(f"Feature index out of range: max_idx={max_idx}, available_features={X_combined.shape[1]}")
        raise ValueError(f"Feature index out of range")

    X_selected = X_combined[:, selected_indices]
    log.info(f"Selected features shape: {X_selected.shape}")

    # Load scaler and transform data
    log.info("Loading scaler and transforming data...")
    scaler = joblib.load(scaler_file)
    X_scaled = scaler.transform(X_selected)

    # Load model
    log.info("Loading Random Forest 18F combined model...")
    model = joblib.load(model_file)

    # Make predictions
    log.info("Making predictions on full B3DB dataset...")
    y_pred = model.predict(X_scaled)

    # Separate samples with and without logBB for evaluation
    has_logbb_mask = df_filtered[pred_column_name].notna()
    n_with_logbb = has_logbb_mask.sum()
    n_without_logbb = (~has_logbb_mask).sum()

    log.info(f"Total predictions: {len(y_pred)}")
    log.info(f"  - With logBB (can evaluate): {n_with_logbb}")
    log.info(f"  - Without logBB (prediction only): {n_without_logbb}")

    # Create output directory
    os.makedirs('./result', exist_ok=True)

    # Save ALL predictions
    log.info("Saving ALL prediction results...")
    results_df = pd.DataFrame({
        'SMILES': df_filtered[smile_column_name].values,
        'Actual_logBB': df_filtered[pred_column_name].values if pred_column_name in df_filtered.columns else [np.nan] * len(df_filtered),
        'Predicted_logBB': y_pred,
        'Has_Actual_Value': has_logbb_mask.values
    })

    # Calculate absolute error only for samples with actual values
    results_df['Absolute_Error'] = np.where(
        results_df['Has_Actual_Value'],
        np.abs(results_df['Actual_logBB'] - results_df['Predicted_logBB']),
        np.nan
    )

    results_df.to_csv('./result/B3DB_RF_18F_Combined_Full_Predictions.csv', index=False)
    log.info("All predictions saved to './result/B3DB_RF_18F_Combined_Full_Predictions.csv'")

    # Calculate metrics only for samples with actual logBB values
    if n_with_logbb > 0:
        y_true = df_filtered.loc[has_logbb_mask, pred_column_name].values
        y_pred_eval = y_pred[has_logbb_mask]

        mse = mean_squared_error(y_true, y_pred_eval)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_eval)
        mae = mean_absolute_error(y_true, y_pred_eval)

        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred_eval) / np.maximum(np.abs(y_true), epsilon))) * 100

        n = len(y_true)
        p = X_selected.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Print and log results
        log.info("\n" + "="*80)
        log.info("External Validation Performance (Samples with logBB values)")
        log.info("="*80)
        log.info(f"Sample count: {n}")
        log.info(f"Feature count: {p}")
        log.info(f"MSE: {mse:.4f}")
        log.info(f"RMSE: {rmse:.4f}")
        log.info(f"R2: {r2:.4f}")
        log.info(f"Adjusted R2: {adj_r2:.4f}")
        log.info(f"MAE: {mae:.4f}")
        log.info(f"MAPE: {mape:.2f}%")

        print("\n" + "="*80)
        print("B3DB External Validation Results (Random Forest 18F Combined)")
        print("="*80)
        print(f"Total samples processed: {len(y_pred)}")
        print(f"  - With logBB (evaluated): {n}")
        print(f"  - Without logBB (predicted only): {n_without_logbb}")
        print(f"Features: {p}")
        print(f"R2 = {r2:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAE = {mae:.4f}")
        print(f"MAPE = {mape:.2f}%")
        print("="*80)

        # Create scatter plot (only for samples with logBB)
        log.info("Creating scatter plot for samples with logBB values...")
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred_eval, alpha=0.6, s=50, edgecolors='black', linewidths=0.5, color='green')

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred_eval.min())
        max_val = max(y_true.max(), y_pred_eval.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='Perfect Prediction')

        plt.xlabel('Experimental logBB', fontsize=14)
        plt.ylabel('Predicted logBB', fontsize=14)
        plt.title('B3DB External Validation - Random Forest 18F Combined Model\n(Morgan + Mordred + Metadata)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Add statistics text box
        textstr = '\n'.join((
            f'N = {n} (with logBB)',
            f'R2 = {r2:.3f}',
            f'Adj. R2 = {adj_r2:.3f}',
            f'RMSE = {rmse:.3f}',
            f'MAE = {mae:.3f}',
            f'MAPE = {mape:.1f}%'
        ))
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plot_file = './result/B3DB_RF_18F_Combined_Full_Scatter.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Scatter plot saved to '{plot_file}'")

        # Save summary report
        import json
        report = {
            'model_info': {
                'model_type': 'Random Forest',
                'features': 'Combined (Morgan Fingerprints + Mordred Descriptors + Metadata)',
                'training_strategy': '18F balanced resampling',
                'model_file': model_file,
                'scaler_file': scaler_file
            },
            'validation_dataset': {
                'name': 'B3DB',
                'file': b3db_file,
                'total_samples': len(y_pred),
                'samples_with_logbb': int(n),
                'samples_without_logbb': int(n_without_logbb),
                'n_features': int(p)
            },
            'performance_metrics': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R2': float(r2),
                'Adjusted_R2': float(adj_r2),
                'MAPE': float(mape)
            }
        }

        report_file = './result/B3DB_RF_18F_Combined_Full_Report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"Validation report saved to '{report_file}'")
    else:
        log.warning("No samples with logBB values found, skipping metric calculation")
        print("\nNo samples with logBB values for evaluation")
        print(f"Total predictions made: {len(y_pred)}")

    log.info("="*80)
    log.info("B3DB Full Dataset Validation Completed")
    log.info("="*80)

if __name__ == "__main__":
    main()
