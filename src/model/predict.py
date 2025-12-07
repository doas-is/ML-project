# inference standalone prediction script

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IVFPredictor:
    """
    Production inference system for IVF patient response prediction.
    
    Loads trained model and provides predictions with explanations.
    """
    
    def __init__(self, model_dir: str = "src/model/saved_models"):
        """
        Initialize predictor by loading saved model and metadata.
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.class_names = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model and metadata."""
        try:
            # Load metadata
            metadata_path = self.model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            self.class_names = self.metadata['class_names']
            best_model_name = self.metadata['best_model']
            
            # Load calibrated model (production-ready)
            model_filename = f"{best_model_name.replace(' ', '_').lower()}_calibrated.pkl"
            model_path = self.model_dir / model_filename
            
            self.model = joblib.load(model_path)
            
            logger.info(f"✓ Loaded model: {best_model_name} (calibrated)")
            logger.info(f"✓ Features: {len(self.feature_names)}")
            logger.info(f"✓ Classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def prepare_features(self, patient_data: dict) -> pd.DataFrame:
        """
        Prepare patient data into model-ready features.
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            DataFrame with features in correct order
        """
        # Required base features
        base_features = {
            'cycle_number': patient_data.get('cycle_number', 1),
            'Age': patient_data.get('age', 30),
            'AMH': patient_data.get('amh', 2.0),
            'n_Follicles': patient_data.get('n_follicles', 10),
            'E2_day5': patient_data.get('e2_day5', 500.0),
            'AFC': patient_data.get('afc', 10),
        }
        
        # Protocol encoding
        protocol_mapping = {
            'flexible antagonist': 0,
            'fixed antagonist': 1,
            'agonist': 2
        }
        protocol = patient_data.get('protocol', 'flexible antagonist').lower()
        base_features['Protocol_encoded'] = protocol_mapping.get(protocol, 0)
        
        # Create derived features (if they exist in model)
        if 'AMH_Age_ratio' in self.feature_names:
            base_features['AMH_Age_ratio'] = base_features['AMH'] / base_features['Age']
        
        if 'AFC_Follicles_ratio' in self.feature_names:
            base_features['AFC_Follicles_ratio'] = base_features['AFC'] / (base_features['n_Follicles'] + 1)
        
        if 'E2_per_follicle' in self.feature_names:
            base_features['E2_per_follicle'] = base_features['E2_day5'] / (base_features['n_Follicles'] + 1)
        
        # Create DataFrame with features in correct order
        # Only include features that the model expects
        feature_dict = {name: base_features.get(name, 0) for name in self.feature_names}
        df = pd.DataFrame([feature_dict])
        
        return df
    
    def predict(self, patient_data: dict, include_explanation: bool = True) -> dict:
        """
        Make prediction for a patient.
        
        Args:
            patient_data: Patient information dictionary
            include_explanation: Include detailed explanation
            
        Returns:
            Dictionary with prediction and probabilities
        """
        # Prepare features
        X = self.prepare_features(patient_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get predicted class
        predicted_label = self.class_names[prediction]
        
        # Format probabilities
        prob_dict = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        # Find max probability
        max_prob_class = max(prob_dict.keys(), key=lambda k: prob_dict[k])
        max_prob = prob_dict[max_prob_class]
        
        # Build result
        result = {
            'predicted_class': int(prediction),
            'predicted_label': predicted_label,
            'confidence': f"{max_prob * 100:.1f}%",
            'probabilities': {k: f"{v * 100:.1f}%" for k, v in prob_dict.items()},
            'probabilities_raw': prob_dict,
            'interpretation': f"{max_prob * 100:.0f}% chance this patient is {max_prob_class} responsive"
        }
        
        # Add clinical recommendation
        if include_explanation:
            result['clinical_recommendation'] = self._generate_recommendation(
                predicted_label, max_prob, patient_data
            )
        
        return result
    
    def _generate_recommendation(self, predicted_label: str, confidence: float, 
                                patient_data: dict) -> str:
        """
        Generate clinical recommendation based on prediction.
        
        Args:
            predicted_label: Predicted response category
            confidence: Prediction confidence
            patient_data: Original patient data
            
        Returns:
            Clinical recommendation string
        """
        recommendations = []
        
        if predicted_label == 'low':
            recommendations.append("⚠️ LOW RESPONSE PREDICTED")
            recommendations.append("• Consider higher starting dose (300+ IU)")
            recommendations.append("• Maximal stimulation protocol recommended")
            recommendations.append("• Close monitoring of follicle development")
            recommendations.append("• May require multiple cycles")
            recommendations.append("• Counsel patient on realistic expectations")
            
        elif predicted_label == 'optimal':
            recommendations.append("✓ OPTIMAL RESPONSE PREDICTED")
            recommendations.append("• Standard protocol appropriate (flexible antagonist)")
            recommendations.append("• Standard starting dose (150-225 IU)")
            recommendations.append("• Regular monitoring schedule")
            recommendations.append("• Good prognosis for cycle success")
            
        elif predicted_label == 'high':
            recommendations.append("⚠️ HIGH RESPONSE PREDICTED - OHSS RISK")
            recommendations.append("• Lower starting dose recommended (100-150 IU)")
            recommendations.append("• Consider antagonist protocol (safer)")
            recommendations.append("• Close E2 monitoring (daily if needed)")
            recommendations.append("• Consider coasting if E2 >3000 pg/mL")
            recommendations.append("• Plan GnRH agonist trigger (vs hCG)")
            recommendations.append("• Consider freeze-all strategy")
        
        # Add confidence note
        if confidence < 0.60:
            recommendations.append(f"\n⚠️ Moderate confidence ({confidence*100:.0f}%)")
            recommendations.append("• Monitor closely during cycle")
            recommendations.append("• Be prepared to adjust protocol")
        elif confidence >= 0.80:
            recommendations.append(f"\n✓ High confidence ({confidence*100:.0f}%)")
        
        return "\n".join(recommendations)
    
    def predict_batch(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple patients.
        
        Args:
            patients_df: DataFrame with patient data
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for idx, row in patients_df.iterrows():
            patient_data = row.to_dict()
            result = self.predict(patient_data, include_explanation=False)
            results.append({
                'patient_index': idx,
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                **result['probabilities']
            })
        
        return pd.DataFrame(results)


def main():
    """
    Command-line interface for predictions.
    """
    parser = argparse.ArgumentParser(
        description='IVF Patient Response Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python predict.py
  
  # Single patient prediction
  python predict.py --age 30 --amh 3.5 --afc 15 --n_follicles 12 --e2_day5 450
  
  # Batch prediction from CSV
  python predict.py --batch patients.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--age', type=int, help='Patient age')
    parser.add_argument('--amh', type=float, help='AMH level (ng/mL)')
    parser.add_argument('--afc', type=int, help='Antral Follicle Count')
    parser.add_argument('--n_follicles', type=int, help='Number of follicles')
    parser.add_argument('--e2_day5', type=float, help='E2 level on day 5 (pg/mL)')
    parser.add_argument('--cycle_number', type=int, default=1, help='IVF cycle number')
    parser.add_argument('--protocol', type=str, default='flexible antagonist',
                       choices=['flexible antagonist', 'fixed antagonist', 'agonist'],
                       help='Stimulation protocol')
    parser.add_argument('--batch', type=str, help='CSV file for batch predictions')
    parser.add_argument('--output', type=str, help='Output file for batch predictions')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  IVF PATIENT RESPONSE PREDICTION SYSTEM")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = IVFPredictor()
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        return
    
    # Batch mode
    if args.batch:
        print(f"\n📊 Batch prediction mode")
        print(f"Input file: {args.batch}")
        
        try:
            df = pd.read_csv(args.batch)
            results = predictor.predict_batch(df)
            
            output_file = args.output or args.batch.replace('.csv', '_predictions.csv')
            results.to_csv(output_file, index=False)
            
            print(f"✓ Predictions saved to: {output_file}")
            print(f"\nPreview:")
            print(results.head().to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error during batch prediction: {str(e)}")
        
        return
    
    # Single patient mode
    if args.age and args.amh:
        # Use provided arguments
        patient_data = {
            'age': args.age,
            'amh': args.amh,
            'afc': args.afc or 10,
            'n_follicles': args.n_follicles or 10,
            'e2_day5': args.e2_day5 or 500,
            'cycle_number': args.cycle_number,
            'protocol': args.protocol
        }
    else:
        # Interactive mode
        print("\n📝 Enter patient information:")
        print("-" * 70)
        
        patient_data = {}
        
        try:
            patient_data['age'] = int(input("Age (years): "))
            patient_data['amh'] = float(input("AMH (ng/mL): "))
            patient_data['afc'] = int(input("AFC (count): "))
            patient_data['n_follicles'] = int(input("Number of Follicles: "))
            patient_data['e2_day5'] = float(input("E2 on day 5 (pg/mL): "))
            patient_data['cycle_number'] = int(input("Cycle Number [1]: ") or "1")
            
            print("\nProtocol options:")
            print("  1. Flexible antagonist")
            print("  2. Fixed antagonist")
            print("  3. Agonist")
            protocol_choice = input("Select protocol [1]: ") or "1"
            
            protocols = {
                '1': 'flexible antagonist',
                '2': 'fixed antagonist',
                '3': 'agonist'
            }
            patient_data['protocol'] = protocols.get(protocol_choice, 'flexible antagonist')
            
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input or cancelled")
            return
    
    # Make prediction
    print("\n" + "="*70)
    print("🔬 ANALYZING PATIENT DATA")
    print("="*70)
    
    try:
        result = predictor.predict(patient_data)
        
        # Display patient info
        print("\n📋 Patient Information:")
        print("-" * 70)
        print(f"Age: {patient_data['age']} years")
        print(f"AMH: {patient_data['amh']:.2f} ng/mL")
        print(f"AFC: {patient_data['afc']}")
        print(f"Number of Follicles: {patient_data['n_follicles']}")
        print(f"E2 (day 5): {patient_data['e2_day5']:.1f} pg/mL")
        print(f"Cycle Number: {patient_data['cycle_number']}")
        print(f"Protocol: {patient_data['protocol']}")
        
        # Display prediction
        print("\n" + "="*70)
        print("🎯 PREDICTION RESULTS")
        print("="*70)
        print(f"\nPredicted Response: {result['predicted_label'].upper()}")
        print(f"Confidence: {result['confidence']}")
        print(f"\n{result['interpretation']}")
        
        print("\n📊 Response Probabilities:")
        print("-" * 70)
        for response, prob in result['probabilities'].items():
            bar_length = int(float(prob.rstrip('%')) / 5)
            bar = "█" * bar_length
            print(f"{response:8} : {bar:20} {prob}")
        
        # Display recommendation
        print("\n" + "="*70)
        print("💊 CLINICAL RECOMMENDATION")
        print("="*70)
        print(result['clinical_recommendation'])
        
        print("\n" + "="*70)
        print("✅ PREDICTION COMPLETE")
        print("="*70)
        print("\n⚠️  Note: This prediction is for decision support only.")
        print("Always validate with clinical judgment and patient-specific factors.\n")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()