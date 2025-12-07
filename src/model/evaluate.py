"""
Model evaluation module with explainability for detailed evaluation metrics, calibration analysis, using SHAP and LIME. It provides :
    - Detailed performance metrics
    - Calibration analysis
    - ROC curves and AUC
    - Confusion matrix analysis
    - SHAP explanations (global & local)
    - LIME explanations (local)
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

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve, auc, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import shap
import lime
import lime.lime_tabular

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model, model_name: str, X_test, y_test, feature_names, class_names):
        self.model = model
        self.model_name = model_name
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Generate predictions
        self.y_pred = model.predict(X_test)

        # Generate predicted probabilities if available
        if hasattr(model, "predict_proba"):
            self.y_pred_proba = model.predict_proba(X_test)
        else:
            self.y_pred_proba = None
            logger.warning(f"{model_name} does not support predict_proba. Brier score will be skipped.")
    
    def calculate_metrics(self):
        logger.info("Calculating evaluation metrics")
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.y_pred, average=None, zero_division=0
        )
        
        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            self.y_test, self.y_pred, average='weighted', zero_division=0
        )
        
        # Brier score (calibration metric) with safety check
        if self.y_pred_proba is not None:
            y_test_binarized = label_binarize(self.y_test, classes=[0, 1, 2])
            brier_scores = []
            for i in range(len(self.class_names)):
                brier = brier_score_loss(y_test_binarized[:, i], self.y_pred_proba[:, i])
                brier_scores.append(brier)
            brier_score_mean = np.mean(brier_scores)
        else:
            brier_score_mean = 0.5  # Default (random guessing level)
            brier_scores = [0.5] * len(self.class_names)
            logger.warning("Brier score set to 0.5 (no probabilities available)")
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece()
        
        # Safety-critical metrics for HIGH response
        y_binary_high = (self.y_test == 2).astype(int)
        y_pred_binary_high = (self.y_pred == 2).astype(int)
        
        # True Negatives and False Positives for specificity
        tn_high = np.sum((y_binary_high == 0) & (y_pred_binary_high == 0))
        fp_high = np.sum((y_binary_high == 0) & (y_pred_binary_high == 1))
        fn_high = np.sum((y_binary_high == 1) & (y_pred_binary_high == 0))
        tp_high = np.sum((y_binary_high == 1) & (y_pred_binary_high == 1))
        
        high_specificity = tn_high / (tn_high + fp_high) if (tn_high + fp_high) > 0 else 0
        high_npv = tn_high / (tn_high + fn_high) if (tn_high + fn_high) > 0 else 0
        high_recall = recall[2]
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_w,
            'recall_weighted': recall_w,
            'f1_weighted': f1_w,
            'brier_score': brier_score_mean,
            'ece': ece,
            'high_recall': high_recall,
            'high_specificity': high_specificity,
            'high_npv': high_npv,
            'per_class': {
                self.class_names[i]: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': int(support[i]),
                    'brier_score': brier_scores[i] if self.y_pred_proba is not None else 0.5
                }
                for i in range(len(self.class_names))
            }
        }
        
        return metrics

    def _calculate_ece(self, n_bins=10):
        """Calculate Expected Calibration Error"""
        if self.y_pred_proba is None:
            logger.warning("ECE cannot be calculated without probabilities")
            return 0.5
        
        ece = 0.0
        for i in range(len(self.class_names)):
            y_binary = (self.y_test == i).astype(int)
            y_prob = self.y_pred_proba[:, i]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_prob, n_bins=n_bins, strategy='uniform'
                )
                
                # Calculate ECE for this class
                bin_weights = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0] / len(y_prob)
                class_ece = np.sum(bin_weights * np.abs(fraction_of_positives - mean_predicted_value))
                ece += class_ece
            except ValueError as e:
                logger.warning(f"ECE calculation failed for class {i}: {e}")
                continue
        
        return ece / len(self.class_names)
    
    def plot_confusion_matrix(self, save_path: str = None):
        logger.info("Generating confusion matrix")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        axes[0].set_title('Confusion matrix (Counts)', fontsize=14, fontweight='bold')
        
        # Percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_roc_curves(self, save_path: str = None):
        if self.y_pred_proba is None:
            logger.warning("ROC curves require probability predictions - skipping")
            return
        
        logger.info("Generating ROC curves")
        
        # Binarize target for multi-class ROC
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_calibration_curves(self, save_path: str = None):
        if self.y_pred_proba is None:
            logger.warning("Calibration curves require probabilities - skipping")
            return
        
        logger.info("Generating calibration curves")
        
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, class_name in enumerate(self.class_names):
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test_bin[:, i], self.y_pred_proba[:, i], n_bins=5
            )
            
            # Plot
            axes[i].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
            axes[i].plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', lw=2, label=f'{class_name}')
            axes[i].set_xlabel('Mean predicted probability', fontsize=11)
            axes[i].set_ylabel('Fraction of positives', fontsize=11)
            axes[i].set_title(f'Calibration: {class_name}', fontsize=12, fontweight='bold')
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved calibration curves to {save_path}")
        
        plt.show()
        plt.close()
    
    def apply_shap_global(self, X_background=None, save_path: str = None):
        logger.info("Generating SHAP global explanations")
        
        if X_background is None:
            X_background = self.X_test
        
        try:
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_background)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_background, feature_names=self.feature_names, class_names=self.class_names, show=False)
            plt.title(f'SHAP Feature Importance - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP summary to {save_path}")
            
            plt.show()
            plt.close()
            
            return shap_values, explainer
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            logger.info("Note: SHAP works best with tree-based models")
            return None, None
    

    def apply_shap_local(self, sample_idx: int, explainer=None, shap_values=None):
        """
        Generate local SHAP explanation for a specific sample
        """
        logger.info(f"Generating SHAP local explanation for sample {sample_idx}")
        
        try:
            if explainer is None:
                explainer = shap.TreeExplainer(self.model)
            
            if shap_values is None:
                shap_values = explainer.shap_values(self.X_test)
            
            # Force plot for specific prediction
            sample = self.X_test.iloc[sample_idx:sample_idx+1]
            pred_class = self.y_pred[sample_idx]
            
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                explainer.expected_value[pred_class],
                shap_values[pred_class][sample_idx],
                sample,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Explanation - Sample {sample_idx} - Predicted: {self.class_names[pred_class]}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
            plt.close()
            
        except Exception as e:
            logger.error(f"SHAP local explanation failed: {str(e)}")
    
    def apply_lime_local(self, sample_idx: int, save_path: str = None):
        logger.info(f"Generating LIME explanation for sample {sample_idx}")
        
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_test.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            
            # Explain instance
            sample = self.X_test.iloc[sample_idx].values
            exp = explainer.explain_instance(
                sample,
                self.model.predict_proba,
                num_features=10
            )
            
            # Display
            fig = exp.as_pyplot_figure()
            plt.title(f'LIME Explanation - Sample {sample_idx}', 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved LIME explanation to {save_path}")
            
            plt.show()
            plt.close()
            
            return exp
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            return None
    
    def generate_medical_interpretation(self, metrics: dict):
        interpretation = []
        
        interpretation.append("="*70)
        interpretation.append("MEDICAL INTERPRETATION OF MODEL PERFORMANCE")
        interpretation.append("="*70)
        
        # 1. OVERALL PERFORMANCE
        accuracy = metrics['accuracy']
        f1_weighted = metrics['f1_weighted']
        brier = metrics['brier_score']
        
        interpretation.append(f"\n1. OVERALL PERFORMANCE (Multi-Metric Assessment):")
        interpretation.append(f"   - Accuracy: {accuracy:.1%}")
        interpretation.append(f"   - Weighted F1-Score: {f1_weighted:.3f}")
        interpretation.append(f"   - Brier Score: {brier:.4f} (calibration)")
        
        # Calculate composite clinical score
        calibration_score = 1 - min(brier / 0.25, 1.0)
        composite_score = (f1_weighted * 0.4) + (accuracy * 0.3) + (calibration_score * 0.3)
        
        interpretation.append(f"   - Composite Clinical Score: {composite_score:.3f}")
        
        # Clinical readiness assessment
        if composite_score >= 0.85 and metrics['high_recall'] >= 0.85:
            interpretation.append("   ✓ CLINICALLY READY - Model meets safety standards")
        elif composite_score >= 0.75 and metrics['high_recall'] >= 0.75:
            interpretation.append("   ✓ ACCEPTABLE - Model suitable with clinical oversight")
        elif composite_score >= 0.65:
            interpretation.append("   ⚠ CAUTION - Requires improvement before clinical deployment")
        else:
            interpretation.append("   ✗ NOT READY - Significant improvement needed")
        
        # 2. SAFETY-CRITICAL METRICS
        interpretation.append("\n2. SAFETY-CRITICAL PERFORMANCE (OHSS Prevention):")
        high_metrics = metrics['per_class']['high']
        
        interpretation.append(f"   High Response Detection:")
        interpretation.append(f"   - Sensitivity (Recall): {high_metrics['recall']:.1%} "
                            f"<- % of actual high responders detected")
        interpretation.append(f"   - Specificity: {metrics.get('high_specificity', 0.0):.1%} "
                            f"<- % of non-high correctly identified")
        interpretation.append(f"   - PPV (Precision): {high_metrics['precision']:.1%} "
                            f"<- Confidence when predicting high")
        interpretation.append(f"   - NPV: {metrics.get('high_npv', 0.0):.1%} "
                            f"<- Confidence when ruling out high")
        
        # OHSS risk assessment
        if high_metrics['recall'] >= 0.90:
            interpretation.append("   ✓ EXCELLENT - Minimal missed OHSS risk cases")
        elif high_metrics['recall'] >= 0.80:
            interpretation.append("   ✓ GOOD - Acceptable OHSS risk detection")
        elif high_metrics['recall'] >= 0.70:
            interpretation.append("   ⚠ MODERATE - Some OHSS cases may be missed")
        else:
            interpretation.append("   ✗ INSUFFICIENT - Too many OHSS cases missed (UNSAFE)")
        
        # 3. EFFICACY METRICS (Low Response Detection)
        interpretation.append("\n3. EFFICACY METRICS (Cycle Success Optimization):")
        low_metrics = metrics['per_class']['low']
        
        interpretation.append(f"   Low Response Detection:")
        interpretation.append(f"   - Sensitivity (Recall): {low_metrics['recall']:.1%} "
                            f"← % of actual low responders detected")
        interpretation.append(f"   - PPV (Precision): {low_metrics['precision']:.1%} "
                            f"← Confidence when predicting low")
        interpretation.append(f"   - F1-Score: {low_metrics['f1']:.3f} "
                            f"← Balance of detection and precision")
        
        if low_metrics['recall'] >= 0.80:
            interpretation.append("   ✓ GOOD - Can effectively identify patients needing dose adjustment")
        elif low_metrics['recall'] >= 0.70:
            interpretation.append("   ⚠ MODERATE - Some low responders may be missed")
        else:
            interpretation.append("   ✗ POOR - Many poor responders not identified")
        
        # 4. OPTIMAL RESPONSE METRICS
        interpretation.append("\n4. OPTIMAL RESPONSE PREDICTION:")
        optimal_metrics = metrics['per_class']['optimal']
        
        interpretation.append(f"   - Precision: {optimal_metrics['precision']:.1%}")
        interpretation.append(f"   - Recall: {optimal_metrics['recall']:.1%}")
        interpretation.append(f"   - F1-Score: {optimal_metrics['f1']:.3f}")
        
        # 5. PROBABILITY CALIBRATION (Critical for clinical decisions)
        interpretation.append(f"\n5. PROBABILITY CALIBRATION:")
        interpretation.append(f"   - Brier Score: {brier:.4f} (lower is better)")
        interpretation.append(f"   - Expected Calibration Error: {metrics.get('ece', 0.0):.4f}")
        
        if brier < 0.10:
            interpretation.append("   ✓ EXCELLENT - Probabilities highly reliable for decisions")
        elif brier < 0.15:
            interpretation.append("   ✓ GOOD - Probabilities suitable for clinical use")
        elif brier < 0.20:
            interpretation.append("   ⚠ ACCEPTABLE - Use probability ranges, not exact values")
        else:
            interpretation.append("   ✗ POOR - Probabilities unreliable, use class only")
        
        # 6. CLINICAL DECISION THRESHOLDS
        interpretation.append("\n6. CLINICAL DECISION ANALYSIS:")
        interpretation.append(f"   Recommended confidence thresholds:")
        
        # Dynamic thresholds based on calibration quality
        if brier < 0.15:
            interpretation.append("   - High confidence: ≥75% (act with confidence)")
            interpretation.append("   - Moderate confidence: 60-75% (standard monitoring)")
            interpretation.append("   - Low confidence: <60% (enhanced clinical assessment)")
        else:
            interpretation.append("   - High confidence: ≥85% (act with confidence)")
            interpretation.append("   - Moderate confidence: 70-85% (standard monitoring)")
            interpretation.append("   - Low confidence: <70% (enhanced clinical assessment)")
        
        # 7. ERROR ANALYSIS
        interpretation.append("\n7. ERROR IMPACT ASSESSMENT:")
        
        # False negatives for high response (most dangerous)
        fn_high = high_metrics['support'] * (1 - high_metrics['recall'])
        if fn_high > 0:
            interpretation.append(f"   ⚠ Missed high responders: ~{fn_high:.1f} cases")
            interpretation.append("     → CRITICAL: These patients face OHSS risk")
        
        # False negatives for low response
        fn_low = low_metrics['support'] * (1 - low_metrics['recall'])
        if fn_low > 0:
            interpretation.append(f"   ⚠ Missed low responders: ~{fn_low:.1f} cases")
            interpretation.append("     → IMPACT: Suboptimal dosing, possible cycle cancellation")
        
        # 8. CLINICAL DEPLOYMENT RECOMMENDATIONS
        interpretation.append("\n8. CLINICAL DEPLOYMENT RECOMMENDATIONS:")
        
        if composite_score >= 0.85 and high_metrics['recall'] >= 0.85:
            interpretation.append("   ✓ APPROVED for clinical decision support")
            interpretation.append("   - Can guide protocol selection and dose optimization")
            interpretation.append("   - Suitable for OHSS risk stratification")
            interpretation.append("   - Recommend quarterly performance audits")
        elif composite_score >= 0.75:
            interpretation.append("   ⚠ CONDITIONAL approval with safeguards:")
            interpretation.append("   - Require clinician review of all predictions")
            interpretation.append("   - Mandatory second opinion for high-risk cases")
            interpretation.append("   - Monthly performance monitoring required")
        else:
            interpretation.append("   ✗ NOT APPROVED for clinical use")
            interpretation.append("   - Model requires improvement")
            interpretation.append("   - Consider additional training data")
            interpretation.append("   - Reassess after retraining")
        
        interpretation.append("\n   General Guidelines:")
        interpretation.append("   - Always combine predictions with clinical judgment")
        interpretation.append("   - Consider patient history and comorbidities")
        interpretation.append("   - Monitor for population drift (annual recalibration)")
        interpretation.append("   - Document all prediction-based decisions")
        
        interpretation.append("\n" + "="*70)
        
        return "\n".join(interpretation)
    
    def generate_full_report(self, save_dir: str = "reports"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("="*70)
        logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        logger.info("="*70)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Generate visualizations
        self.plot_confusion_matrix(f"{save_dir}/confusion_matrix_detailed.png")
        self.plot_roc_curves(f"{save_dir}/roc_curves.png")
        self.plot_calibration_curves(f"{save_dir}/calibration_curves.png")
        
        # SHAP analysis
        shap_values, explainer = self.apply_shap_global(
            save_path=f"{save_dir}/shap_summary.png"
        )
        
        # LIME for first sample
        if len(self.X_test) > 0:
            self.apply_lime_local(0, f"{save_dir}/lime_sample_0.png")
        
        # Medical interpretation
        medical_interp = self.generate_medical_interpretation(metrics)
        
        # Save text report
        report_path = f"{save_dir}/evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"MODEL EVALUATION REPORT - {self.model_name}\n")
            f.write("="*70 + "\n\n")
            
            f.write("METRICS SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Weighted Precision: {metrics['precision_weighted']:.4f}\n")
            f.write(f"Weighted Recall: {metrics['recall_weighted']:.4f}\n")
            f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n")
            f.write(f"Brier Score: {metrics['brier_score']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-"*70 + "\n")
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"\n{class_name.upper()}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n")
                f.write(f"  Brier Score: {class_metrics['brier_score']:.4f}\n")
            
            f.write("\n\n")
            f.write(medical_interp)
        
        logger.info(f"Report saved to {report_path}")
        logger.info("="*70)
        
        return metrics, medical_interp


def main():
    print("\n" + "="*70)
    print("  MODEL EVALUATION")
    
    # Load test data and model
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "patients_cleaned.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model", "saved_models")
    REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not Path(DATA_PATH).exists():
        print(f"❌ Error: Data folder not found at {DATA_PATH}")
        return
    
    try:
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Prepare features (same as training)
        exclude_cols = ['patient_id', 'Patient Response', 'Protocol', 'Age_group', 'AMH_category','Patient Response_encoded']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        target_col = 'Patient Response_encoded' if 'Patient Response_encoded' in df.columns else 'Patient Response'
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split (same as training for consistency)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Load metadata
        with open(f"{MODEL_DIR}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        best_model_name = metadata['best_model']
        feature_names = metadata['feature_names']
        class_names = metadata['class_names']
        
        # Load best model (calibrated)
        model_filename = f"{best_model_name.replace(' ', '_').lower()}_calibrated.pkl"
        model = joblib.load(f"{MODEL_DIR}/{model_filename}")
        
        print(f"\n✓ Loaded model: {best_model_name} (calibrated)")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model, best_model_name, X_test, y_test,
            feature_names, class_names
        )
        
        # Generate full report
        metrics, medical_interp = evaluator.generate_full_report(REPORT_DIR)
        
        # Display results
        print("\n" + medical_interp)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📊 Reports saved to: {REPORT_DIR}/")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()