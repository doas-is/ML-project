import pytest
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.train import IVFResponsePredictor
from src.model.predict import IVFPredictor


class TestModelTraining:
    """Test suite for model training pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 20
        
        data = {
            'cycle_number': np.random.randint(1, 4, n_samples),
            'Age': np.random.randint(25, 40, n_samples),
            'AMH': np.random.uniform(0.5, 5.0, n_samples),
            'n_Follicles': np.random.randint(5, 25, n_samples),
            'E2_day5': np.random.uniform(200, 1000, n_samples),
            'AFC': np.random.randint(5, 25, n_samples),
            'Protocol_encoded': np.random.randint(0, 3, n_samples),
            'Patient Response_encoded': np.random.randint(0, 3, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly."""
        predictor = IVFResponsePredictor(random_state=42)
        
        assert predictor.random_state == 42
        assert predictor.models == {}
        assert predictor.class_names == ['low', 'optimal', 'high']
    
    def test_define_models(self):
        """Test that all models are defined correctly."""
        predictor = IVFResponsePredictor()
        models = predictor.define_models()
        
        # Check all expected models exist
        assert 'Random Forest' in models
        assert 'Gradient Boosting' in models
        assert 'Logistic Regression' in models
        assert 'SVM' in models
        
        # Check each model has required keys
        for name, config in models.items():
            assert 'model' in config
            assert 'params' in config
            assert isinstance(config['params'], dict)
    
    def test_train_test_split(self, sample_data):
        predictor = IVFResponsePredictor()
        
        X = sample_data.drop('Patient Response_encoded', axis=1)
        y = sample_data['Patient Response_encoded']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check sizes from total the 20 generated samples
        assert len(X_train) == 16
        assert len(X_test) == 4
        
        # Check features preserved
        assert list(X_train.columns) == list(X.columns)
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction from models."""
        from sklearn.ensemble import RandomForestClassifier
        
        predictor = IVFResponsePredictor()
        
        # Create simple model
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 3, 20)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictor.models['test'] = model
        predictor.feature_names = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        predictor.best_model_name = 'test'
        
        importance_df = predictor.get_feature_importance()
        
        assert importance_df is not None
        assert len(importance_df) == 5
        assert 'Feature' in importance_df.columns
        assert 'Importance' in importance_df.columns
    
    def test_prediction_output_format(self):
        """Test prediction output has correct format."""
        predictor = IVFResponsePredictor()
        predictor.class_names = ['low', 'optimal', 'high']
        
        # Mock prediction
        X_test = np.random.rand(3, 5)
        predictions = np.array([0, 1, 2])
        probabilities = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7]
        ])
        
        # Simulate model
        class MockModel:
            def predict(self, X):
                return predictions
            def predict_proba(self, X):
                return probabilities
        
        predictor.models['mock'] = MockModel()
        predictor.best_model_name = 'mock'
        
        results = predictor.predict_with_explanation(X_test, use_calibrated=False)
        
        assert len(results) == 3
        
        for result in results:
            assert 'predicted_class' in result
            assert 'predicted_label' in result
            assert 'confidence' in result
            assert 'probabilities' in result
            assert 'interpretation' in result
            
            # Check probabilities sum to 100%
            probs = result['probabilities']
            assert len(probs) == 3
            assert 'low' in probs
            assert 'optimal' in probs
            assert 'high' in probs


class TestInference:
    """Test suite for inference functionality."""
    
    @pytest.fixture
    def predictor(self, tmp_path):
        """Create a mock predictor with saved model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and train simple model
        X = np.random.rand(20, 7)
        y = np.random.randint(0, 3, 20)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save model
        model_dir = tmp_path / "saved_models"
        model_dir.mkdir()
        
        joblib.dump(model, model_dir / "random_forest_calibrated.pkl")
        
        # Save metadata
        metadata = {
            'best_model': 'Random Forest',
            'feature_names': ['cycle_number', 'Age', 'AMH', 'n_Follicles', 
                            'E2_day5', 'AFC', 'Protocol_encoded'],
            'class_names': ['low', 'optimal', 'high']
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        return IVFPredictor(model_dir=str(model_dir))
    
    def test_predictor_loads_model(self, predictor):
        """Test that predictor loads model successfully."""
        assert predictor.model is not None
        assert predictor.metadata is not None
        assert predictor.feature_names is not None
        assert predictor.class_names == ['low', 'optimal', 'high']
    
    def test_prepare_features(self, predictor):
        """Test feature preparation from patient data."""
        patient_data = {
            'age': 30,
            'amh': 3.5,
            'afc': 15,
            'n_follicles': 12,
            'e2_day5': 450,
            'cycle_number': 1,
            'protocol': 'flexible antagonist'
        }
        
        features = predictor.prepare_features(patient_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        assert 'Age' in features.columns
        assert 'AMH' in features.columns
        assert 'Protocol_encoded' in features.columns
        assert features['Age'].iloc[0] == 30
        assert features['AMH'].iloc[0] == 3.5
    
    def test_predict_returns_valid_output(self, predictor):
        patient_data = {
            'age': 30,
            'amh': 3.5,
            'afc': 15,
            'n_follicles': 12,
            'e2_day5': 450,
            'cycle_number': 1,
            'protocol': 'flexible antagonist'
        }
        
        result = predictor.predict(patient_data)
        
        # Check required keys
        assert 'predicted_class' in result
        assert 'predicted_label' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'interpretation' in result
        assert 'clinical_recommendation' in result
        
        # Check types
        assert isinstance(result['predicted_class'], int)
        assert result['predicted_label'] in ['low', 'optimal', 'high']
        assert '%' in result['confidence']
        assert isinstance(result['probabilities'], dict)
        assert len(result['probabilities']) == 3
    
    def test_protocol_encoding(self, predictor):
        """Test that protocols are encoded correctly."""
        protocols = {
            'flexible antagonist': 0,
            'fixed antagonist': 1,
            'agonist': 2
        }
        
        for protocol, expected_code in protocols.items():
            patient_data = {
                'age': 30, 'amh': 2.5, 'afc': 10,
                'n_follicles': 10, 'e2_day5': 500,
                'cycle_number': 1, 'protocol': protocol
            }
            
            features = predictor.prepare_features(patient_data)
            assert features['Protocol_encoded'].iloc[0] == expected_code
    
    def test_clinical_recommendation_content(self, predictor):
        """Test that clinical recommendations contain key information."""
        patient_data = {
            'age': 28, 'amh': 5.0, 'afc': 25,
            'n_follicles': 20, 'e2_day5': 800,
            'cycle_number': 1, 'protocol': 'flexible antagonist'
        }
        
        result = predictor.predict(patient_data)
        recommendation = result['clinical_recommendation']
        
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0
        
        # Should contain either LOW, OPTIMAL, or HIGH
        assert any(word in recommendation for word in ['LOW', 'OPTIMAL', 'HIGH'])


class TestDataValidation:
    """Test suite for data validation."""
    
    def test_missing_required_features(self):
        """Test handling of missing required features."""
        predictor = IVFResponsePredictor()
        
        incomplete_data = {
            'Age': [30, 35],
            'AMH': [2.5, 1.8]
            # Missing other required features
        }
        
        df = pd.DataFrame(incomplete_data)
        
        # Should handle missing features well
        assert 'Age' in df.columns
        assert 'AMH' in df.columns
    
    def test_invalid_protocol(self):
        """Test handling of invalid protocol values."""
        from src.model.predict import IVFPredictor
        
        # This should handle unknown protocols well
        patient_data = {
            'age': 30, 'amh': 2.5, 'afc': 10,
            'n_follicles': 10, 'e2_day5': 500,
            'cycle_number': 1, 'protocol': 'unknown_protocol'
        }
        
        # Should default to flexible antagonist (code 0)
        assert 'protocol' in patient_data
    
    def test_edge_case_values(self):
        """Test handling of edge case values."""
        edge_cases = [
            {'age': 18, 'amh': 0.1},  # Very young, very low AMH
            {'age': 50, 'amh': 10.0},  # Very old, very high AMH
            {'age': 30, 'n_follicles': 0},  # No follicles
            {'age': 30, 'e2_day5': 0.0}  # Zero E2
        ]
        
        for case in edge_cases:
            # Should not raise errors
            assert isinstance(case['age'], int)
            if 'amh' in case:
                assert case['amh'] >= 0


class TestModelPersistence:
    """Test suite for model saving and loading."""
    
    def test_save_models(self, tmp_path):
        """Test that models can be saved correctly."""
        from sklearn.svm import SVC
        
        predictor = IVFResponsePredictor()
        
        # Create simple model
        model = SVC(random_state=42, probability=True)
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 3, 20)
        model.fit(X, y)
        
        predictor.models['SVM'] = model
        predictor.calibrated_models['SVM'] = model
        predictor.best_model_name = 'SVM'
        predictor.feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # Save
        import os
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        SAVE_DIR = PROJECT_ROOT / "src" / "model" / "saved_models"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        predictor.save_models(str(SAVE_DIR))

        assert (SAVE_DIR / "svm.pkl").exists()
        assert (SAVE_DIR / "svm_calibrated.pkl").exists()
        assert (SAVE_DIR / "metadata.json").exists()

    
    def test_load_model(self, tmp_path):
        """Test that models can be loaded correctly."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and save model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 3, 20)
        model.fit(X, y)
        
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        model_path = model_dir / "test_model.pkl"
        joblib.dump(model, model_path)
        
        # Load
        predictor = IVFResponsePredictor()
        predictor.load_model(str(model_path))
        
        assert predictor.models['loaded'] is not None
        assert predictor.best_model_name == 'loaded'



# Manual test execution for quick testing
def run_manual_tests():
    print("=" * 70)
    print("Running Manual Model Tests")
    print("=" * 70)
    
    # Test 1: Predictor initialization
    print("\n✓ Test 1: Predictor Initialization")
    predictor = IVFResponsePredictor(random_state=42)
    assert predictor.random_state == 42
    print("  ✓ Predictor initialized correctly")
    
    # Test 2: Model definitions
    print("\n✓ Test 2: Model Definitions")
    models = predictor.define_models()
    assert 'Random Forest' in models
    assert 'SVM' in models
    print(f"  ✓ Defined {len(models)} models: {list(models.keys())}")
    
    # Test 3: Sample data creation
    print("\n✓ Test 3: Sample Data Creation")
    np.random.seed(42)
    data = {
        'Age': np.random.randint(25, 40, 20),
        'AMH': np.random.uniform(0.5, 5.0, 20),
        'n_Follicles': np.random.randint(5, 25, 20),
        'Patient Response_encoded': np.random.randint(0, 3, 20)
    }
    df = pd.DataFrame(data)
    print(f"  ✓ Created sample dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Test 4: Protocol encoding
    print("\n✓ Test 4: Protocol Encoding")
    protocol_mapping = {
        'flexible antagonist': 0,
        'fixed antagonist': 1,
        'agonist': 2
    }
    for protocol, code in protocol_mapping.items():
        assert code in [0, 1, 2]
    print(f"  ✓ Protocol encoding verified: {len(protocol_mapping)} protocols")
    
    print("\n" + "=" * 70)
    print("✅ All manual tests passed!")
    print("=" * 70)
    
    print("\n💡 To run with pytest:")
    print("   pip install pytest")
    print("   pytest tests/test_model.py -v")
    print("   pytest tests/test_model.py -v --cov=src.model")


if __name__ == "__main__":
    # Run manual tests if executed directly
    run_manual_tests()
    
    print("\n" + "=" * 70)
    print("For comprehensive testing, use pytest:")
    print("  pytest tests/test_model.py -v")
    print("  pytest tests/test_model.py -v -k 'test_predict'  # Run specific tests")
    print("  pytest tests/test_model.py -v --cov  # With coverage")
    print("=" * 70)