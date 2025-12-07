"""
Probabilistic classification model for IVF patient response prediction:
    - Multiple model training (RF, GBM, LogReg)
    - Probability calibration
    - Model selection via cross-validation
    - Explainability (SHAP/LIME ready)
    - Medical interpretation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, log_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import label_binarize 
"""label_binarize converts multi-class labels to one-hot encoded format
its needed for multi-class ROC curves or AUC.""" 

import logging

logging.basicConfig( level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IVFResponsePredictor:  
    def __init__(self, random_state: int = 42): # seed for reproducibility
        self.random_state = random_state
        self.models = {}
        self.calibrated_models = {}
        self.best_model_name = None
        self.feature_names = None
        self.class_names = ['low', 'optimal', 'high']
        self.results = {}
        
    def load_and_prepare_data(self, file_path: str, test_size: float = 0.2):
        
        # Load preprocessed data and split into train/test & returns a tuple of (X_train, X_test, y_train, y_test)
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Define features to use

        # Drop patient_id and original categorical columns 
        exclude_cols = ['patient_id', 'Patient Response', 'Protocol', 'Age_group', 'AMH_category','Patient Response_encoded']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle encoded target
        if 'Patient Response_encoded' in df.columns:
            target_col = 'Patient Response_encoded'
        else:
            target_col = 'Patient Response'
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Stratified split to ensure that train and test sets have the same class distribution as the original dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def define_models(self):
        # candidate models with each its hyperparameters
        logger.info("Defining candidate models")
        
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': Categorical([70, 100, 200]),
                    'max_depth': Categorical([3, 5, 7, None]),
                    'min_samples_split': Integer(2, 5),
                    'min_samples_leaf': Integer(1, 2),
                    'class_weight': Categorical(['balanced', None])
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': Integer(50, 150),
                    'learning_rate': Categorical([0.05, 0.1, 0.2]),
                    'max_depth': Integer(3, 7),
                    'min_samples_split': Integer(2, 5),
                    'subsample': Real(0.8, 1.0)
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='multinomial'
                ),
                'params': {
                    'C': Real(0.01, 10.0, prior='log-uniform'),
                    'penalty': Categorical(['l2']),
                    'class_weight': Categorical(['balanced', None])
                }
            },
            'SVM': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': Real(0.1, 10.0, prior='log-uniform'),
                    'kernel': Categorical(['rbf', 'linear']),
                    'gamma': Categorical(['scale', 'auto']),
                    'class_weight': Categorical(['balanced', None])
                }
            }
        }

        return models
    
    def train_model_with_cv(self, model, params, X_train, y_train, cv_folds: int = 5):
        # Use stratified k-fold since it s a relatively small dataset with class imbalance
        logger.info(f"Training model with Bayesian Optimization and {cv_folds}-fold CV")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # BayesSearchCV for hyperparameter tuning
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=params,
            cv=cv,
            scoring='f1_weighted', # Good for multi-class
            n_jobs=-1,
            verbose=0,
            random_state=self.random_state
        )
        
        bayes_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {bayes_search.best_params_}")
        logger.info(f"Best CV score: {bayes_search.best_score_:.4f}")
        
        return bayes_search.best_estimator_
    
    def calibrate_model(self, model, X_train, y_train, method='sigmoid'):
        # we're using CalibratedClassifierCV for model calibration
        logger.info(f"Calibrating model with method: {method}")
        
        calibrated = CalibratedClassifierCV(model, method=method, cv=3)
        calibrated.fit(X_train, y_train)
        
        return calibrated
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        logger.info("TRAINING ALL MODELS")
        logger.info("="*70)
        
        models_config = self.define_models()
        results = {}
        
        for name, config in models_config.items():
            logger.info(f"\nTraining {name}...")
            
            try:
                # Train with CV + Bayesian optimization
                best_model = self.train_model_with_cv(
                    config['model'], 
                    config['params'],
                    X_train, 
                    y_train,
                    cv_folds=5
                )
                
                self.models[name] = best_model
                
                # Calibrate probabilities
                calibrated_model = self.calibrate_model(best_model, X_train, y_train)
                self.calibrated_models[name] = calibrated_model
                
                # Predict on test set
                y_pred = best_model.predict(X_test)
                
                if hasattr(best_model, "predict_proba"):
                    y_pred_proba = best_model.predict_proba(X_test)
                else:
                    y_pred_proba = calibrated_model.predict_proba(X_test)
                
                # Evaluate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision_weighted': precision_score(
                        y_test, y_pred, average='weighted', zero_division=0),
                    'recall_weighted': recall_score(
                        y_test, y_pred, average='weighted', zero_division=0),
                    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'log_loss': log_loss(y_test, y_pred_proba),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }

                # Compute minority recall
                try:
                    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    minority_recall = min(
                        class_report[c]['recall']
                        for c in class_report if c not in ['accuracy', 'macro avg', 'weighted avg']
                    )
                    metrics['minority_recall'] = minority_recall
                except:
                    metrics['minority_recall'] = 0

                results[name] = metrics

                logger.info(
                    f"✓ {name} | F1_w: {metrics['f1_weighted']:.4f} | "
                    f"LogLoss: {metrics['log_loss']:.4f} | "
                    f"MinorityRecall: {metrics['minority_recall']:.4f}"
                )

            except Exception as e:
                logger.error(f"✗ {name} training failed: {str(e)}")
                continue
        
        # ← UNINDENT THIS SECTION (move it out of the for loop)
        # Multi-criteria model selection
        def score_model(m):
            F1 = results[m]['f1_weighted']
            MR = results[m].get('minority_recall', 0)
            LL = results[m]['log_loss']
            
            # Metrics combo: higher F1 is better, lower log-loss is better
            return (F1 * 0.6) + (MR * 0.3) + (-LL * 0.1)
        
        best_name = max(results.keys(), key=score_model)
        self.best_model_name = best_name
        self.best_model = self.calibrated_models.get(best_name, self.models[best_name])
        self.results = results
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏆 BEST MODEL: {best_name}")
        logger.info(f"F1 Score: {results[best_name]['f1_weighted']:.4f}")
        logger.info(f"Log-Loss: {results[best_name]['log_loss']:.4f}")
        logger.info(f"Minority Recall: {results[best_name]['minority_recall']:.4f}")
        logger.info(f"{'='*70}")
        
        return results

    # Multi-criteria model selection
    def score_model(m):
        F1 = results[m]['f1_weighted']
        MR = results[m].get('minority_recall', 0)
        LL = results[m]['log_loss']

        # Metrics combo: higher F1 is better, lower log-loss is penalized
        return (F1 * 0.6) + (MR * 0.3) + (-LL * 0.1)

        best_name = max(results.keys(), key=score_model)
        self.best_model_name = best_name
        self.best_model = self.calibrated_models.get(best_name, self.models[best_name])
        self.results = results

        logger.info(f"\n{'=' *70}")
        logger.info(f"🏆 BEST MODEL: {best_name}")
        logger.info(f"F1 Score: {results[best_name]['f1_weighted']:.4f}")
        logger.info(f"Log-Loss: {results[best_name]['log_loss']:.4f}")
        logger.info(f"Minority Recall: {results[best_name]['minority_recall']}")
        logger.info(f"{'=' *70}")

        return results
    
    def get_feature_importance(self, model_name: str = None):
        # Fix 1: Check if best_model_name exists
        if model_name is None:
            if self.best_model_name is None:
                logger.warning("No best model selected yet")
                return None
            model_name = self.best_model_name
        
        # Fix 2: Check if model exists in dictionary
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found in trained models")
            return None
        
        model = self.models[model_name]
        
        # Fix 3: Check if feature_names exist
        if self.feature_names is None:
            logger.warning("Feature names not available")
            return None
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For logistic regression, use absolute coefficients
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            logger.warning(f"{model_name} does not have feature importances")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def predict_with_explanation(self, X, model_name: str = None, use_calibrated: bool = True):
        # check best_model_name exists
        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("No model has been trained yet")
            model_name = self.best_model_name
        
        # check model exists before accessing
        if use_calibrated and model_name in self.calibrated_models:
            model = self.calibrated_models[model_name]
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Format output
        results = []
        for i in range(len(predictions)):
            pred_class = predictions[i]
            probs = probabilities[i]
            
            # Find class with highest probability
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]
            max_class = self.class_names[max_prob_idx]
            
            result = {
                'predicted_class': pred_class,
                'predicted_label': self.class_names[pred_class],
                'confidence': f"{max_prob*100:.1f}%",
                'probabilities': {
                    'low': f"{probs[0]*100:.1f}%",
                    'optimal': f"{probs[1]*100:.1f}%",
                    'high': f"{probs[2]*100:.1f}%"
                },
                'interpretation': f"{max_prob*100:.0f}% chance this patient is {max_class} responsive"
            }
            results.append(result)
        
        return results
    
    def save_models(self, save_dir: str = "src/model/saved_models"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # validate before saving
        if not self.models:
            logger.warning("No models to save")
            return
        
        if self.best_model_name is None:
            logger.warning("No best model selected, cannot save metadata")
            return
    
        # all models
        for name, model in self.models.items():
            model_path = Path(save_dir) / f"{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # calibrated models
        for name, model in self.calibrated_models.items():
            model_path = Path(save_dir) / f"{name.replace(' ', '_').lower()}_calibrated.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved calibrated {name} to {model_path}")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'random_state': self.random_state
        }
        
        metadata_path = Path(save_dir) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    

    def load_model(self, model_path: str, metadata_path: str = None):
        self.models['loaded'] = joblib.load(model_path)
        self.best_model_name = 'loaded'
        
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
                self.class_names = metadata['class_names']
        
        logger.info(f"Loaded model from {model_path}")




def visualize_results(predictor, X_test, y_test, save_dir: str):
    # Create comprehensive visualizations of model results

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance
    logger.info("Generating feature importance plot")
    importance_df = predictor.get_feature_importance()
    
    if importance_df is not None:
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top 10 Feature Importance - {predictor.best_model_name}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Confusion Matrix
    logger.info("Generating confusion matrix")
    if predictor.best_model_name is None:
        logger.error("No best model selected")
        return
    
    if predictor.best_model_name not in predictor.results:
        logger.error(f"Results for {predictor.best_model_name} not found")
        return
    
    # Now safe to access
    y_pred = predictor.results[predictor.best_model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=predictor.class_names,
                yticklabels=predictor.class_names)
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {predictor.best_model_name}', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model Comparison
    logger.info("Generating model comparison")
    results_df = pd.DataFrame({
        name: {
            'Accuracy': res['accuracy'],
            'F1-Score': res['f1_weighted'],
            'Precision': res['precision_weighted'],
            'Recall': res['recall_weighted']
        }
        for name, res in predictor.results.items()
    }).T
    
    results_df.plot(kind='bar', figsize=(12, 6), rot=0)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()






def main():
    print("\n" + "="*70)
    print("  IVF PATIENT RESPONSE PREDICTION - MODEL TRAINING")
    print("="*70)

    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "patients_cleaned.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model", "saved_models")
    figures_dir = os.path.join(PROJECT_ROOT, "src", "model", "models_figures")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not Path(DATA_PATH).exists():
        print(f"❌ Error: Data folder not found at {DATA_PATH}")
        return
    
    try:
        # Initialize predictor
        predictor = IVFResponsePredictor(random_state=42)
        
        # Load and split data
        X_train, X_test, y_train, y_test = predictor.load_and_prepare_data(DATA_PATH, test_size=0.2)
        
        # Train all models
        results = predictor.train_all_models(X_train, X_test, y_train, y_test)
        
        # Display detailed results
        print("\nMODEL PERFORMANCE SUMMARY\n")
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_weighted']:.4f}")
            print(f"  Recall:    {metrics['recall_weighted']:.4f}")
            print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
            print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        
        # Feature importance
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE (Top 10)")
        print("="*70)
        importance_df = predictor.get_feature_importance()
        if importance_df is not None:
            print(importance_df.head(10).to_string(index=False))
        
        # Example predictions
        print("\n" + "="*70)
        print("EXAMPLE PREDICTIONS (First 3 Test Samples)")
        print("="*70)
        
        predictions = predictor.predict_with_explanation(X_test[:3], use_calibrated=True)
        for i, pred in enumerate(predictions):
            print(f"\nSample {i+1}:")
            print(f"  Prediction: {pred['predicted_label']}")
            print(f"  Interpretation: {pred['interpretation']}")
            print(f"  Probabilities:")
            for cls, prob in pred['probabilities'].items():
                print(f"    {cls}: {prob}")
        
        # Visualizations
        visualize_results(predictor, X_test, y_test, figures_dir)
        
        # Save models
        predictor.save_models(MODEL_DIR)
        
        # Success
        print("\n✅ MODEL TRAINING COMPLETED SUCCESSFULLY!\n")
        print("="*70)
        print(f"\n📊 Best Model: {predictor.best_model_name}")
        print(f"📁 Models saved to: {MODEL_DIR}")
        print(f"📈 Figures saved to: {figures_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()