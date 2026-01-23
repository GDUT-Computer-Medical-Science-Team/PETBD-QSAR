"""
PETBD LogBB Prediction Script
==============================
Standalone script for predicting blood-brain barrier permeability (logBB)
using pre-trained models from the PETBD-QSAR project.

Usage:
    # Single SMILES prediction
    python predict_logBB.py --smiles "CCO" --model xgboost
    
    # Batch prediction from CSV
    python predict_logBB.py --input compounds.csv --output predictions.csv --model xgboost
    
    # List available models
    python predict_logBB.py --list-models

Requirements:
    - Python 3.8+
    - numpy, pandas, scikit-learn, rdkit, joblib
    - xgboost (for XGBoost models)
    - lightgbm (for LightGBM models)
    - catboost (for CatBoost models)
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("Error: RDKit is required but not installed.")
    print("Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


class LogBBPredictor:
    """Predictor class for logBB using PETBD models"""
    
    AVAILABLE_MODELS = {
        'xgboost': {
            'model_file': 'MachineLearningModels/logBBModel/logbbModel/xgb_fp_18F_model.joblib',
            'scaler_file': 'MachineLearningModels/logBBModel/logbbModel/xgb_fp_18F_scaler.joblib',
            'features_file': 'MachineLearningModels/logBBModel/logbbModel/xgb_fp_18F_features.npy',
            'description': 'XGBoost model with Morgan fingerprints (18F resampled)'
        },
        'rf': {
            'model_file': 'MachineLearningModels/logBBModel/logbbModel/rf_fp_18F_model.joblib',
            'scaler_file': 'MachineLearningModels/logBBModel/logbbModel/rf_fp_18F_scaler.joblib',
            'features_file': 'MachineLearningModels/logBBModel/logbbModel/rf_fp_18F_features.npy',
            'description': 'Random Forest model with Morgan fingerprints (18F resampled)'
        },
        'lightgbm': {
            'model_file': 'MachineLearningModels/logBBModel/logbbModel/lightgbm_petbd_18F_model.joblib',
            'scaler_file': None,
            'features_file': None,
            'description': 'LightGBM model (18F resampled)'
        }
    }
    
    def __init__(self, model_name='xgboost', repo_root='.'):
        """
        Initialize predictor
        
        Args:
            model_name: Name of model to use ('xgboost', 'rf', 'lightgbm')
            repo_root: Path to repository root directory
        """
        self.model_name = model_name
        self.repo_root = Path(repo_root)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_config = self.AVAILABLE_MODELS[model_name]
        self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and feature indices"""
        # Load model
        model_path = self.repo_root / self.model_config['model_file']
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please ensure you have trained the model or downloaded pre-trained models."
            )
        
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        
        # Load scaler (if exists)
        if self.model_config['scaler_file']:
            scaler_path = self.repo_root / self.model_config['scaler_file']
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded scaler from: {scaler_path}")
            else:
                print(f"Warning: Scaler file not found: {scaler_path}")
                self.scaler = None
        else:
            self.scaler = None
        
        # Load feature indices (if exists)
        if self.model_config['features_file']:
            features_path = self.repo_root / self.model_config['features_file']
            if features_path.exists():
                self.feature_indices = np.load(features_path)
                print(f"Loaded {len(self.feature_indices)} feature indices")
            else:
                print(f"Warning: Feature indices file not found: {features_path}")
                self.feature_indices = None
        else:
            self.feature_indices = None
        
        print(f"Model '{self.model_name}' loaded successfully!\n")
    
    @staticmethod
    def calculate_morgan_fingerprint(smiles, radius=2, n_bits=1024):
        """
        Calculate Morgan fingerprint from SMILES
        
        Args:
            smiles: SMILES string
            radius: Morgan fingerprint radius (default: 2)
            n_bits: Number of bits (default: 1024)
            
        Returns:
            numpy array of fingerprint bits
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES: {smiles}")
            return np.zeros(n_bits)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    def predict(self, smiles_list):
        """
        Predict logBB for a list of SMILES
        
        Args:
            smiles_list: List of SMILES strings or single SMILES string
            
        Returns:
            numpy array of predictions
        """
        # Handle single SMILES
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Calculate fingerprints
        print(f"Calculating molecular fingerprints for {len(smiles_list)} compounds...")
        features = np.array([self.calculate_morgan_fingerprint(s) for s in smiles_list])
        
        # Select features if indices provided
        if self.feature_indices is not None:
            features = features[:, self.feature_indices]
            print(f"Selected {features.shape[1]} features")
        
        # Scale features if scaler provided
        if self.scaler is not None:
            features = self.scaler.transform(features)
            print("Applied feature scaling")
        
        # Predict
        print("Making predictions...")
        predictions = self.model.predict(features)
        
        return predictions
    
    def predict_from_csv(self, input_csv, output_csv=None, smiles_column='SMILES'):
        """
        Predict logBB from CSV file
        
        Args:
            input_csv: Path to input CSV file (must contain SMILES column)
            output_csv: Path to output CSV file (optional)
            smiles_column: Name of SMILES column (default: 'SMILES')
            
        Returns:
            DataFrame with predictions
        """
        # Read input
        print(f"Reading input from: {input_csv}")
        df = pd.read_csv(input_csv)
        
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Predict
        predictions = self.predict(df[smiles_column].tolist())
        
        # Add predictions to dataframe
        df['Predicted_logBB'] = predictions
        
        # Classify BBB permeability
        df['BBB_Permeability'] = df['Predicted_logBB'].apply(
            lambda x: 'High' if x > 0 else ('Moderate' if x > -0.5 else 'Low')
        )
        
        # Save output
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Predict blood-brain barrier permeability (logBB) using PETBD models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single compound
  python predict_logBB.py --smiles "CCO" --model xgboost
  
  # Batch prediction
  python predict_logBB.py --input compounds.csv --output predictions.csv
  
  # List available models
  python predict_logBB.py --list-models
        """
    )
    
    parser.add_argument('--smiles', type=str, help='Single SMILES string to predict')
    parser.add_argument('--input', type=str, help='Input CSV file with SMILES column')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    parser.add_argument('--smiles-column', type=str, default='SMILES', 
                       help='Name of SMILES column in input CSV (default: SMILES)')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=list(LogBBPredictor.AVAILABLE_MODELS.keys()),
                       help='Model to use for prediction (default: xgboost)')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Path to repository root (default: current directory)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 80)
        for name, config in LogBBPredictor.AVAILABLE_MODELS.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")
            print(f"  Model file: {config['model_file']}")
        print("\n")
        return
    
    # Validate arguments
    if not args.smiles and not args.input:
        parser.error("Either --smiles or --input must be provided")
    
    if args.input and not args.output:
        print("Warning: No output file specified. Results will be displayed only.")
    
    # Initialize predictor
    try:
        predictor = LogBBPredictor(model_name=args.model, repo_root=args.repo_root)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you are in the repository root directory")
        print("2. Check that model files exist in MachineLearningModels/logBBModel/logbbModel/")
        print("3. If models don't exist, train them first or download pre-trained models")
        sys.exit(1)
    
    # Predict single SMILES
    if args.smiles:
        print(f"\nPredicting logBB for: {args.smiles}")
        print("-" * 80)
        
        prediction = predictor.predict(args.smiles)[0]
        
        print(f"\nResults:")
        print(f"  SMILES: {args.smiles}")
        print(f"  Predicted logBB: {prediction:.4f}")
        
        if prediction > 0:
            permeability = "High"
            desc = "Good BBB permeability (logBB > 0)"
        elif prediction > -0.5:
            permeability = "Moderate"
            desc = "Moderate BBB permeability (-0.5 < logBB < 0)"
        else:
            permeability = "Low"
            desc = "Poor BBB permeability (logBB < -0.5)"
        
        print(f"  BBB Permeability: {permeability}")
        print(f"  Interpretation: {desc}")
        print()
    
    # Predict from CSV
    elif args.input:
        try:
            results = predictor.predict_from_csv(
                args.input, 
                args.output, 
                smiles_column=args.smiles_column
            )
            
            print("\nPrediction Summary:")
            print("-" * 80)
            print(f"Total compounds: {len(results)}")
            print(f"\nBBB Permeability Distribution:")
            print(results['BBB_Permeability'].value_counts())
            print(f"\nStatistics:")
            print(f"  Mean logBB: {results['Predicted_logBB'].mean():.4f}")
            print(f"  Std logBB: {results['Predicted_logBB'].std():.4f}")
            print(f"  Min logBB: {results['Predicted_logBB'].min():.4f}")
            print(f"  Max logBB: {results['Predicted_logBB'].max():.4f}")
            
            if not args.output:
                print("\nFirst 10 predictions:")
                print(results[[args.smiles_column, 'Predicted_logBB', 'BBB_Permeability']].head(10).to_string(index=False))
            
            print()
            
        except Exception as e:
            print(f"\nError during prediction: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

