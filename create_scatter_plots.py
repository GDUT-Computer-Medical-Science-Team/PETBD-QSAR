"""
Create comprehensive scatter plots for best performing models
Using actual CSV data for train, validation, and test sets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for better Chinese support (if needed)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_model_scatter_plot(model_name, dataset_type, base_path):
    """
    Create scatter plot for a specific model showing all three datasets
    """
    
    # Define file paths
    if dataset_type == 'logBB':
        prefix = 'petbd'
        target_label = 'logBB'
        unit = ''
    else:
        prefix = 'cbrain'
        target_label = 'Brain Concentration'
        unit = ' (60min)'
    
    # Load prediction CSV files
    train_file = f"{base_path}/result/{model_name}_{prefix}_18F_train_predictions.csv"
    val_file = f"{base_path}/result/{model_name}_{prefix}_18F_val_predictions.csv"
    test_file = f"{base_path}/result/{model_name}_{prefix}_18F_test_predictions.csv"
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        print(f"Missing prediction files for {model_name} - {dataset_type}")
        return None
    
    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Calculate metrics
    train_r2 = r2_score(train_df['True_Values'], train_df['Predicted_Values'])
    train_rmse = np.sqrt(mean_squared_error(train_df['True_Values'], train_df['Predicted_Values']))
    
    val_r2 = r2_score(val_df['True_Values'], val_df['Predicted_Values'])
    val_rmse = np.sqrt(mean_squared_error(val_df['True_Values'], val_df['Predicted_Values']))
    
    test_r2 = r2_score(test_df['True_Values'], test_df['Predicted_Values'])
    test_rmse = np.sqrt(mean_squared_error(test_df['True_Values'], test_df['Predicted_Values']))
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors and sizes
    colors = {'train': '#3498db', 'val': '#2ecc71', 'test': '#e74c3c'}
    
    # Plot 1: Training Set
    ax1 = axes[0]
    ax1.scatter(train_df['True_Values'], train_df['Predicted_Values'], 
               alpha=0.6, s=50, color=colors['train'], edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    lims = [min(train_df['True_Values'].min(), train_df['Predicted_Values'].min()),
            max(train_df['True_Values'].max(), train_df['Predicted_Values'].max())]
    ax1.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=1.5)
    
    # Add regression line
    z = np.polyfit(train_df['True_Values'], train_df['Predicted_Values'], 1)
    p = np.poly1d(z)
    ax1.plot(sorted(train_df['True_Values']), p(sorted(train_df['True_Values'])), 
            "r-", alpha=0.5, linewidth=2)
    
    ax1.set_xlabel(f'Experimental {target_label}{unit}', fontsize=12)
    ax1.set_ylabel(f'Predicted {target_label}{unit}', fontsize=12)
    ax1.set_title(f'Training Set (n={len(train_df)})\nR² = {train_r2:.4f}, RMSE = {train_rmse:.4f}', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Set
    ax2 = axes[1]
    ax2.scatter(val_df['True_Values'], val_df['Predicted_Values'], 
               alpha=0.6, s=50, color=colors['val'], edgecolors='black', linewidth=0.5)
    
    lims = [min(val_df['True_Values'].min(), val_df['Predicted_Values'].min()),
            max(val_df['True_Values'].max(), val_df['Predicted_Values'].max())]
    ax2.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=1.5)
    
    z = np.polyfit(val_df['True_Values'], val_df['Predicted_Values'], 1)
    p = np.poly1d(z)
    ax2.plot(sorted(val_df['True_Values']), p(sorted(val_df['True_Values'])), 
            "r-", alpha=0.5, linewidth=2)
    
    ax2.set_xlabel(f'Experimental {target_label}{unit}', fontsize=12)
    ax2.set_ylabel(f'Predicted {target_label}{unit}', fontsize=12)
    ax2.set_title(f'Validation Set (n={len(val_df)})\nR² = {val_r2:.4f}, RMSE = {val_rmse:.4f}', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test Set
    ax3 = axes[2]
    ax3.scatter(test_df['True_Values'], test_df['Predicted_Values'], 
               alpha=0.6, s=50, color=colors['test'], edgecolors='black', linewidth=0.5)
    
    lims = [min(test_df['True_Values'].min(), test_df['Predicted_Values'].min()),
            max(test_df['True_Values'].max(), test_df['Predicted_Values'].max())]
    ax3.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=1.5)
    
    z = np.polyfit(test_df['True_Values'], test_df['Predicted_Values'], 1)
    p = np.poly1d(z)
    ax3.plot(sorted(test_df['True_Values']), p(sorted(test_df['True_Values'])), 
            "r-", alpha=0.5, linewidth=2)
    
    ax3.set_xlabel(f'Experimental {target_label}{unit}', fontsize=12)
    ax3.set_ylabel(f'Predicted {target_label}{unit}', fontsize=12)
    ax3.set_title(f'Test Set (n={len(test_df)})\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}', 
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Overall title
    model_display_name = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lightgbm': 'LightGBM'}
    dataset_display_name = {'logBB': 'LogBB (PETBD)', 'cbrain': 'Cbrain (OrganDataAt60min)'}
    
    fig.suptitle(f'{model_display_name.get(model_name, model_name)} - {dataset_display_name.get(dataset_type, dataset_type)}\n18F Resampled Model Performance', 
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    save_path = f"{model_name}_{dataset_type}_scatter_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def create_combined_best_models_plot():
    """
    Create a combined plot showing the best models' performance
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = [
        ('rf', 'logBB', 'MachineLearningModels/logBBModel'),
        ('xgb', 'logBB', 'MachineLearningModels/logBBModel'),
        ('lightgbm', 'logBB', 'MachineLearningModels/logBBModel'),
        ('rf', 'cbrain', 'MachineLearningModels/CbrainModel'),
        ('xgb', 'cbrain', 'MachineLearningModels/CbrainModel'),
        ('lightgbm', 'cbrain', 'MachineLearningModels/CbrainModel')
    ]
    
    model_display_names = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'lightgbm': 'LightGBM'}
    
    for idx, ((model_name, dataset_type, base_path), ax) in enumerate(zip(models, axes.flat)):
        
        # Define prefix
        prefix = 'petbd' if dataset_type == 'logBB' else 'cbrain'
        
        # Load test predictions only for combined plot
        test_file = f"{base_path}/result/{model_name}_{prefix}_18F_test_predictions.csv"
        
        if not os.path.exists(test_file):
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            continue
        
        test_df = pd.read_csv(test_file)
        
        # Calculate metrics
        test_r2 = r2_score(test_df['True_Values'], test_df['Predicted_Values'])
        test_rmse = np.sqrt(mean_squared_error(test_df['True_Values'], test_df['Predicted_Values']))
        
        # Create scatter plot
        ax.scatter(test_df['True_Values'], test_df['Predicted_Values'], 
                  alpha=0.7, s=60, color=f'C{idx}', edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        lims = [test_df['True_Values'].min(), test_df['True_Values'].max()]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=1.5)
        
        # Add regression line
        z = np.polyfit(test_df['True_Values'], test_df['Predicted_Values'], 1)
        p = np.poly1d(z)
        ax.plot(sorted(test_df['True_Values']), p(sorted(test_df['True_Values'])), 
               "r-", alpha=0.5, linewidth=2)
        
        # Labels
        target_label = 'logBB' if dataset_type == 'logBB' else 'Cbrain'
        ax.set_xlabel(f'Experimental {target_label}', fontsize=11)
        ax.set_ylabel(f'Predicted {target_label}', fontsize=11)
        ax.set_title(f'{model_display_names[model_name]} - {dataset_type.upper()}\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'n = {len(test_df)}\nSlope = {z[0]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    # Overall title
    fig.suptitle('Best Performing Models - Test Set Performance\n18F Resampled Experiments', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    save_path = "combined_best_models_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def create_performance_comparison_bar_plot():
    """
    Create bar plot comparing R² and RMSE for all models
    """
    
    models_data = []
    
    # Collect data for all models
    model_configs = [
        ('rf', 'logBB', 'MachineLearningModels/logBBModel', 'RF-LogBB'),
        ('xgb', 'logBB', 'MachineLearningModels/logBBModel', 'XGB-LogBB'),
        ('lightgbm', 'logBB', 'MachineLearningModels/logBBModel', 'LGBM-LogBB'),
        ('rf', 'cbrain', 'MachineLearningModels/CbrainModel', 'RF-Cbrain'),
        ('xgb', 'cbrain', 'MachineLearningModels/CbrainModel', 'XGB-Cbrain'),
        ('lightgbm', 'cbrain', 'MachineLearningModels/CbrainModel', 'LGBM-Cbrain')
    ]
    
    for model_name, dataset_type, base_path, label in model_configs:
        prefix = 'petbd' if dataset_type == 'logBB' else 'cbrain'
        test_file = f"{base_path}/result/{model_name}_{prefix}_18F_test_predictions.csv"
        
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file)
            r2 = r2_score(test_df['True_Values'], test_df['Predicted_Values'])
            rmse = np.sqrt(mean_squared_error(test_df['True_Values'], test_df['Predicted_Values']))
            models_data.append({'Model': label, 'R²': r2, 'RMSE': rmse})
    
    # Create DataFrame
    df = pd.DataFrame(models_data)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    bars1 = ax1.bar(df['Model'], df['R²'], color=['#3498db', '#e74c3c', '#2ecc71']*2, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance Comparison - R² Score', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars1, df['R²']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE comparison
    bars2 = ax2.bar(df['Model'], df['RMSE'], color=['#9b59b6', '#f39c12', '#1abc9c']*2,
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Model Performance Comparison - RMSE', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars2, df['RMSE']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Overall title
    fig.suptitle('18F Resampled Models - Performance Metrics', 
                fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    # Save figure
    save_path = "model_performance_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig

def main():
    """
    Main function to create all scatter plots
    """
    
    print("=" * 80)
    print("CREATING COMPREHENSIVE SCATTER PLOTS FOR BEST MODELS")
    print("=" * 80)
    print()
    
    # Create individual plots for each model
    print("Creating individual model scatter plots...")
    print("-" * 40)
    
    # LogBB models
    create_model_scatter_plot('rf', 'logBB', 'MachineLearningModels/logBBModel')
    create_model_scatter_plot('xgb', 'logBB', 'MachineLearningModels/logBBModel')
    create_model_scatter_plot('lightgbm', 'logBB', 'MachineLearningModels/logBBModel')
    
    # Cbrain models
    create_model_scatter_plot('rf', 'cbrain', 'MachineLearningModels/CbrainModel')
    create_model_scatter_plot('xgb', 'cbrain', 'MachineLearningModels/CbrainModel')
    create_model_scatter_plot('lightgbm', 'cbrain', 'MachineLearningModels/CbrainModel')
    
    print()
    print("Creating combined plots...")
    print("-" * 40)
    
    # Create combined plot
    create_combined_best_models_plot()
    
    # Create performance comparison
    create_performance_comparison_bar_plot()
    
    print()
    print("=" * 80)
    print("ALL SCATTER PLOTS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - Individual model scatter plots (6 files)")
    print("  - combined_best_models_scatter.png")
    print("  - model_performance_comparison.png")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()