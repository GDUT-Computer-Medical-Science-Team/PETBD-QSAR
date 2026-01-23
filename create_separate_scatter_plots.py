"""
Create separate scatter plots for each dataset (train, validation, test)
For best performing models (RF and XGBoost) on both LogBB and Cbrain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def create_single_scatter_plot(data_df, dataset_name, model_name, target_type, metrics, save_name):
    """
    Create a single scatter plot for one dataset
    """
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define colors for each dataset type
    colors = {
        'Train': '#3498db',
        'Validation': '#2ecc71', 
        'Test': '#e74c3c'
    }
    
    # Get color based on dataset name
    color = colors.get(dataset_name, '#3498db')
    
    # Create scatter plot
    ax.scatter(data_df['True_Values'], data_df['Predicted_Values'],
              alpha=0.6, s=80, color=color, edgecolors='black', linewidth=0.8,
              label=f'{dataset_name} Data')
    
    # Add diagonal line (perfect prediction)
    lims = [min(data_df['True_Values'].min(), data_df['Predicted_Values'].min()),
            max(data_df['True_Values'].max(), data_df['Predicted_Values'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(data_df['True_Values'], data_df['Predicted_Values'], 1)
    p = np.poly1d(z)
    x_sorted = np.sort(data_df['True_Values'])
    ax.plot(x_sorted, p(x_sorted), "r-", alpha=0.7, linewidth=2.5, 
           label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # Labels and title
    if target_type == 'logBB':
        xlabel = 'Experimental logBB'
        ylabel = 'Predicted logBB'
        title_prefix = 'LogBB Prediction'
    else:
        xlabel = 'Experimental Brain Concentration (60min)'
        ylabel = 'Predicted Brain Concentration (60min)'
        title_prefix = 'Cbrain Prediction'
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    model_display = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lightgbm': 'LightGBM'}
    ax.set_title(f'{model_display.get(model_name, model_name)} - {title_prefix}\n{dataset_name} Set',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add metrics text box
    textstr = f'n = {len(data_df)}\n' \
             f'R² = {metrics["r2"]:.4f}\n' \
             f'RMSE = {metrics["rmse"]:.4f}\n' \
             f'MAE = {metrics["mae"]:.4f}'
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props, fontweight='bold')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Set axis limits with padding
    x_range = data_df['True_Values'].max() - data_df['True_Values'].min()
    y_range = data_df['Predicted_Values'].max() - data_df['Predicted_Values'].min()
    
    ax.set_xlim(data_df['True_Values'].min() - 0.1*x_range, 
               data_df['True_Values'].max() + 0.1*x_range)
    ax.set_ylim(data_df['Predicted_Values'].min() - 0.1*y_range,
               data_df['Predicted_Values'].max() + 0.1*y_range)
    
    # Make the plot square
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_name}")
    
    return fig

def process_model(model_name, dataset_type, base_path):
    """
    Process a single model and create scatter plots for all datasets
    """
    
    # Define file paths
    if dataset_type == 'logBB':
        prefix = 'petbd'
        target_type = 'logBB'
    else:
        prefix = 'cbrain'
        target_type = 'cbrain'
    
    # File paths
    train_file = f"{base_path}/result/{model_name}_{prefix}_18F_train_predictions.csv"
    val_file = f"{base_path}/result/{model_name}_{prefix}_18F_val_predictions.csv"
    test_file = f"{base_path}/result/{model_name}_{prefix}_18F_test_predictions.csv"
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        print(f"Missing prediction files for {model_name} - {dataset_type}")
        return
    
    print(f"\nProcessing {model_name.upper()} - {dataset_type}")
    print("-" * 40)
    
    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    datasets = [
        (train_df, 'Train', train_file),
        (val_df, 'Validation', val_file),
        (test_df, 'Test', test_file)
    ]
    
    for df, dataset_name, _ in datasets:
        # Calculate metrics
        r2 = r2_score(df['True_Values'], df['Predicted_Values'])
        rmse = np.sqrt(mean_squared_error(df['True_Values'], df['Predicted_Values']))
        mae = mean_absolute_error(df['True_Values'], df['Predicted_Values'])
        
        metrics = {'r2': r2, 'rmse': rmse, 'mae': mae}
        
        # Create save name
        save_name = f"{model_name}_{dataset_type}_{dataset_name.lower()}_scatter.png"
        
        # Create plot
        create_single_scatter_plot(df, dataset_name, model_name, target_type, metrics, save_name)

def create_summary_comparison_plot():
    """
    Create a summary plot comparing all models and datasets
    """
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Models and datasets to plot
    models = [
        ('rf', 'logBB', 'MachineLearningModels/logBBModel'),
        ('xgb', 'logBB', 'MachineLearningModels/logBBModel'),
        ('rf', 'cbrain', 'MachineLearningModels/CbrainModel'),
        ('xgb', 'cbrain', 'MachineLearningModels/CbrainModel')
    ]
    
    datasets = ['Train', 'Validation', 'Test']
    colors = {'Train': '#3498db', 'Validation': '#2ecc71', 'Test': '#e74c3c'}
    
    for col_idx, (model_name, dataset_type, base_path) in enumerate(models):
        prefix = 'petbd' if dataset_type == 'logBB' else 'cbrain'
        
        for row_idx, dataset_name in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            
            # Load data
            file_path = f"{base_path}/result/{model_name}_{prefix}_18F_{dataset_name.lower()}_predictions.csv"
            
            if not os.path.exists(file_path):
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
                ax.set_title(f'{model_name.upper()}-{dataset_type}\n{dataset_name}')
                continue
            
            df = pd.read_csv(file_path)
            
            # Calculate metrics
            r2 = r2_score(df['True_Values'], df['Predicted_Values'])
            rmse = np.sqrt(mean_squared_error(df['True_Values'], df['Predicted_Values']))
            
            # Plot
            ax.scatter(df['True_Values'], df['Predicted_Values'],
                      alpha=0.6, s=30, color=colors[dataset_name], 
                      edgecolors='black', linewidth=0.5)
            
            # Diagonal line
            lims = [df['True_Values'].min(), df['True_Values'].max()]
            ax.plot(lims, lims, 'k--', alpha=0.75, linewidth=1)
            
            # Title and labels
            model_display = {'rf': 'RF', 'xgb': 'XGB'}
            ax.set_title(f'{model_display.get(model_name, model_name)}-{dataset_type}\n{dataset_name} (R²={r2:.3f})',
                        fontsize=10)
            
            if row_idx == 2:  # Bottom row
                ax.set_xlabel('True Values', fontsize=9)
            if col_idx == 0:  # Left column
                ax.set_ylabel('Predicted Values', fontsize=9)
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Model Comparison - All Datasets\n18F Resampled Models', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('summary_all_models_datasets.png', dpi=300, bbox_inches='tight')
    print("Saved: summary_all_models_datasets.png")
    
    return fig

def main():
    """
    Main function to create all scatter plots
    """
    
    print("=" * 80)
    print("CREATING SEPARATE SCATTER PLOTS FOR EACH DATASET")
    print("=" * 80)
    
    # Process LogBB models
    print("\n=== LogBB Models (PETBD Dataset) ===")
    process_model('rf', 'logBB', 'MachineLearningModels/logBBModel')
    process_model('xgb', 'logBB', 'MachineLearningModels/logBBModel')
    
    # Process Cbrain models
    print("\n=== Cbrain Models (OrganDataAt60min Dataset) ===")
    process_model('rf', 'cbrain', 'MachineLearningModels/CbrainModel')
    process_model('xgb', 'cbrain', 'MachineLearningModels/CbrainModel')
    
    # Create summary plot
    print("\n=== Creating Summary Comparison Plot ===")
    create_summary_comparison_plot()
    
    print("\n" + "=" * 80)
    print("ALL SCATTER PLOTS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  RF LogBB: rf_logBB_train_scatter.png, rf_logBB_validation_scatter.png, rf_logBB_test_scatter.png")
    print("  XGBoost LogBB: xgb_logBB_train_scatter.png, xgb_logBB_validation_scatter.png, xgb_logBB_test_scatter.png")
    print("  RF Cbrain: rf_cbrain_train_scatter.png, rf_cbrain_validation_scatter.png, rf_cbrain_test_scatter.png")
    print("  XGBoost Cbrain: xgb_cbrain_train_scatter.png, xgb_cbrain_validation_scatter.png, xgb_cbrain_test_scatter.png")
    print("  Summary: summary_all_models_datasets.png")

if __name__ == "__main__":
    main()