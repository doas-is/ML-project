"""
    This module handles
    - Initial data validation
    - Missing value analysis and imputation
    - Outlier detection and treatment
    - Feature scaling and normalization
    - Categorical encoding for ML readiness
    - Train-test split preparation
    """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IVFDataPreprocessor:    
    def __init__(self):
        # default parameters
        self.scaler = None
        self.label_encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    # Add this method to the IVFDataPreprocessor class (around line 100, after load_data method):

    def encode_patient_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding patient IDs")
        df = df.copy()
        
        # create mapping: row_number -> 25XXX format
        for idx, row_num in enumerate(range(1, len(df) + 1), start=1):
            df.loc[idx-1, 'patient_id'] = f"25{row_num:03d}"
        
        logger.info(f"Encoded {len(df)} patient IDs (2501 to 25{len(df):03d})")
        return df

    def initial_data_validation(self, df: pd.DataFrame) -> Dict:
        # initial data quality checks.
        logger.info("Performing initial data validation")
        
        validation = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'duplicate_records': df.duplicated().sum(),
            'missing_summary': {},
            'data_types': df.dtypes.to_dict()
        }
        
        # Missing value analysis
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            validation['missing_summary'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        
        # Log results
        logger.info(f"Total records: {validation['total_records']}")
        logger.info(f"Duplicate records: {validation['duplicate_records']}")
        
        missing_cols = [col for col, info in validation['missing_summary'].items() 
                       if info['count'] > 0]
        if missing_cols:
            logger.warning(f"Columns with missing values: {missing_cols}")
        
        return validation
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, 
                           multiplier: float = 1.5) -> Tuple[np.ndarray, Dict]:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1- multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        outlier_info = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(df)) * 100
        }
        
        return outlier_mask, outlier_info
    
    def detect_outliers_all_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        
        logger.info("Detecting outliers in numeric features...")
        
        outlier_report = {}

        for col in numeric_cols:
            col_series = df[col].dropna()

            if col_series.empty: # No numeric values at all
                continue

            # Call the existing IQR function
            mask, info = self.detect_outliers_iqr(df, col)

            # indices for later cleaning
            outlier_indices = df.index[mask].tolist()
            info["outlier_indices"] = outlier_indices

            outlier_report[col] = info

            # Logging
            if info["n_outliers"] > 0:
                logger.warning(
                    f"{col}: {info['n_outliers']} outliers "
                    f"({info['outlier_percentage']:.1f}%)"
                )

        logger.info("Outlier detection completed.")
        return outlier_report

    
    def handle_outliers(self, df: pd.DataFrame, column: str, method: str = 'clip') -> pd.DataFrame:
        """
        There will be different ways to treat the outliers by specified method.
            'clip' : replaces outliers by the IQR lower or upper bounds
            'cap'   : caps the outliers at 99th percentile
        we havent defined remove cuz data in medical records is precious
        """
        outlier_mask, info = self.detect_outliers_iqr(df, column)
        
        if method == 'clip':
            # Clip values to IQR bounds
            df[column] = df[column].clip(
                lower=info['lower_bound'],
                upper=info['upper_bound']
            )
            logger.info(f"{column}: Clipped {info['n_outliers']} outliers")
            
        elif method == 'cap':
            # Cap at 99th percentile to keep the skewed shape of data
            upper_cap = df[column].quantile(0.99)
            df.loc[df[column] > upper_cap, column] = upper_cap
            logger.info(f"{column}: Capped outliers at 99th percentile")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        logger.info("Handling missing values")
        df = df.copy()
        
        for col, method in strategy.items(): #imputing method per column
            if col not in df.columns:
                continue
                
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
            
            if method == 'mean':
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]])
                self.imputers[col] = imputer
                logger.info(f"{col}: Imputed {missing_count} values with mean")
                
            elif method == 'median':
                imputer = SimpleImputer(strategy='median')
                df[col] = imputer.fit_transform(df[[col]])
                self.imputers[col] = imputer
                logger.info(f"{col}: Imputed {missing_count} values with median")
                
            elif method == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]].values.reshape(-1, 1)).ravel()
                self.imputers[col] = imputer
                logger.info(f"{col}: Imputed {missing_count} values with mode")
                
            elif method == 'knn':
                # Use KNN imputation for correlated features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                knn_imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
                self.imputers[col] = knn_imputer
                logger.info(f"{col}: Imputed {missing_count} values with KNN")
                
            elif method == 'drop':
                df = df.dropna(subset=[col])
                logger.info(f"{col}: Dropped {missing_count} rows with missing values")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        
        logger.info("Encoding categorical features")
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
                # Log encoding mapping
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                logger.info(f"{col} encoding: {mapping}")
        return df
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        #scale numeric features using StandardScaler
        
        logger.info("Scaling numeric features")
        df = df.copy()
        
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Store statistics
        for i, col in enumerate(numeric_cols):
            self.feature_stats[col] = {
                'mean': self.scaler.mean_[i],
                'std': self.scaler.scale_[i]
            }
            logger.info(f"{col}: μ={self.feature_stats[col]['mean']:.3f}, "
                       f"σ={self.feature_stats[col]['std']:.3f}")
        
        return df




    """ based on research, I discovered that we can create some derived features that will help go beyond raw values 
    and that are useful for patient stratification when building our ML model
    """

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating derived features")
        df = df.copy()

        # AMH to Age ratio (ovarian reserve relative to age)
        if 'AMH' in df.columns and 'Age' in df.columns:
            df['AMH_Age_ratio'] = df['AMH'] / df['Age']
            logger.info("Created feature: AMH_Age_ratio")
        
        # AFC to Follicles ratio (efficiency)
        if 'AFC' in df.columns and 'n_Follicles' in df.columns:
            df['AFC_Follicles_ratio'] = df['AFC'] / (df['n_Follicles'] + 1)
            logger.info("Created feature: AFC_Follicles_ratio")
        
        # E2 per follicle (hormone production efficiency)
        if 'E2_day5' in df.columns and 'n_Follicles' in df.columns:
            df['E2_per_follicle'] = df['E2_day5'] / (df['n_Follicles'] + 1)
            logger.info("Created feature: E2_per_follicle")
        
        # Age groups (clinical categories)
        if 'Age' in df.columns:
            df['Age_group'] = pd.cut(df['Age'], 
                                    bins=[0, 30, 35, 40, 100],
                                    labels=['<30', '30-35', '35-40', '>40'])
            logger.info("Created feature: Age_group")
        
        # AMH categories (ovarian reserve status)
        if 'AMH' in df.columns:
            df['AMH_category'] = pd.cut(df['AMH'],
                                       bins=[0, 1.0, 3.5, 100],
                                       labels=['Low', 'Normal', 'High'])
            logger.info("Created feature: AMH_category")
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        logger.info("=" * 70)
        logger.info("Starting Complete Preprocessing Pipeline")
        logger.info("=" * 70)
        
        report = {}
        
        # 1: Initial validation
        report['initial_validation'] = self.initial_data_validation(df)
        
        # 2: Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        report['duplicates_removed'] = initial_len - len(df)
        logger.info(f"Removed {report['duplicates_removed']} duplicate records")
        df = self.encode_patient_ids(df)
        
        # 3: Handle missing values
        # defined imputation strategy based on clinical knowledge
        imputation_strategy = {
            'AFC': 'median',  # AFC often missing, use median
            'E2_day5': 'median',
            'AMH': 'median',
            'n_Follicles': 'median',
            'Age': 'median',
            'cycle_number': 'mode'
        }
        df = self.handle_missing_values(df, imputation_strategy)
        
        # 4: Detect outliers (before handling)
        numeric_cols = ['Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        report['outlier_detection'] = self.detect_outliers_all_features(df, numeric_cols)
        
        # 5: Handle outliers (clip extreme values)
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                df = self.handle_outliers(df, col, method='clip')
        
        # 6: Create derived features
        df = self.create_derived_features(df)
        
        # 7: Encode categorical features
        categorical_cols = ['Protocol', 'Patient Response']
        df = self.encode_categorical_features(df, categorical_cols)
        
        # 8: Final validation
        report['final_validation'] = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isna().sum().to_dict()
        }
        
        logger.info("=" * 70)
        logger.info("Preprocessing Pipeline Completed")
        logger.info(f"Final dataset: {len(df)} records, {len(df.columns)} features")
        logger.info("=" * 70)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Preprocessed data saved to {save_path}")
        
        return df, report
    
    def generate_preprocessing_report(self, report: Dict, output_path: str = "preprocessing_report.txt"):
        # a detailed report of preprocessing steps
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("IVF PATIENT DATA PREPROCESSING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Initial validation
            f.write("INITIAL DATA VALIDATION\n")
            f.write("-" * 70 + "\n")
            init_val = report['initial_validation']
            f.write(f"Total Records: {init_val['total_records']}\n")
            f.write(f"Total Features: {init_val['total_features']}\n")
            f.write(f"Duplicate Records: {init_val['duplicate_records']}\n\n")
            
            f.write("Missing Values Summary:\n")
            for col, info in init_val['missing_summary'].items():
                if info['count'] > 0:
                    f.write(f"  {col}: {info['count']} ({info['percentage']:.1f}%)\n")
            f.write("\n")
            
            # Outlier detection
            f.write("OUTLIER DETECTION\n")
            f.write("-" * 70 + "\n")
            for col, info in report['outlier_detection'].items():
                f.write(f"{col}:\n")
                f.write(f"  Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n")
                f.write(f"  Outliers: {info['n_outliers']} ({info['outlier_percentage']:.1f}%)\n\n")
            
            # Final validation
            f.write("FINAL DATA VALIDATION\n")
            f.write("-" * 70 + "\n")
            final_val = report['final_validation']
            f.write(f"Total Records: {final_val['total_records']}\n")
            f.write(f"Total Features: {final_val['total_features']}\n")
            f.write(f"Duplicates Removed: {report['duplicates_removed']}\n")
            
        logger.info(f"Preprocessing report saved to {output_path}")


def main():
    print("\n" + "=" * 70)
    print("  IVF PATIENT DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    

    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Data folders
    PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed")

    # Input and output paths
    input_path = os.path.join(PROCESSED_DATA, "patients_updated.csv")
    output_path = os.path.join(PROCESSED_DATA, "patients_cleaned.csv")
    report_path = os.path.join(PROCESSED_DATA, "preprocessing_report.txt") # for explainability
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"❌ Error: Input file not found at {input_path}")
        print("Please run PDF extraction first.")
        return
    
    try:
        preprocessor = IVFDataPreprocessor()
        df = preprocessor.load_data(input_path)
        
        # Display initial info
        print("\n📊 Initial Dataset Info:")
        print(f"  Records: {len(df)}")
        print(f"  Features: {len(df.columns)}")
        print(f"  Columns: {', '.join(df.columns)}")
        
        # Run preprocessing pipeline
        df_cleaned, report = preprocessor.preprocess_pipeline(df, output_path)
        
        # Generate report
        preprocessor.generate_preprocessing_report(report, report_path)
        
        # Display summary
        print("\n✅ Preprocessing Completed Successfully!")
        print(f"📁 Cleaned data: {output_path}")
        print(f"📄 Report: {report_path}")
        print(f"📊 Final dataset: {len(df_cleaned)} records, {len(df_cleaned.columns)} features")
        
        # Display sample
        print("\n📋 Sample of Preprocessed Data (first 3 rows):")
        print(df_cleaned.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()