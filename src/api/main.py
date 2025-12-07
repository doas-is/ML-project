from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict import IVFPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IVF Patient Response Prediction API",
    description="RESTful API for predicting IVF patient response to ovarian stimulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - CONFIGURE FOR PRODUCTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit
        "http://localhost:3000",  # React dev
        "*"  # Remove in production!
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[IVFPredictor] = None


# PYDANTIC MODELS

class PatientData(BaseModel):
    """Patient data input model with validation."""
    age: int = Field(
        ..., 
        ge=18, 
        le=50, 
        description="Patient age in years (18-50)",
        example=30
    )
    amh: float = Field(
        ..., 
        ge=0.0, 
        le=15.0, 
        description="AMH level in ng/mL (0-15)",
        example=3.5
    )
    afc: int = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Antral Follicle Count (0-50)",
        example=15
    )
    n_follicles: int = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Number of follicles at monitoring",
        example=12
    )
    e2_day5: float = Field(
        ..., 
        ge=0.0, 
        le=5000.0, 
        description="Estradiol level on day 5 in pg/mL",
        example=450.0
    )
    cycle_number: int = Field(
        default=1, 
        ge=1, 
        le=10, 
        description="IVF cycle attempt number",
        example=1
    )
    protocol: str = Field(
        default="flexible antagonist",
        description="Stimulation protocol type",
        example="flexible antagonist"
    )
    
    @validator('protocol')
    def validate_protocol(cls, v):
        allowed = ['flexible antagonist', 'fixed antagonist', 'agonist']
        if v.lower() not in allowed:
            raise ValueError(f"Protocol must be one of: {', '.join(allowed)}")
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "amh": 3.5,
                "afc": 15,
                "n_follicles": 12,
                "e2_day5": 450.0,
                "cycle_number": 1,
                "protocol": "flexible antagonist"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    
    predicted_class: int = Field(..., description="Predicted class (0=low, 1=optimal, 2=high)")
    predicted_label: str = Field(..., description="Predicted response label")
    confidence: str = Field(..., description="Prediction confidence percentage")
    probabilities: Dict[str, str] = Field(..., description="Probability for each class")
    interpretation: str = Field(..., description="Human-readable interpretation")
    clinical_recommendation: str = Field(..., description="Clinical guidance")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_class": 2,
                "predicted_label": "high",
                "confidence": "72.3%",
                "probabilities": {
                    "low": "8.5%",
                    "optimal": "19.2%",
                    "high": "72.3%"
                },
                "interpretation": "72% chance this patient is high responsive",
                "clinical_recommendation": "⚠️ HIGH RESPONSE PREDICTED - OHSS RISK \n ..."
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    model_loaded: bool
    model_functional: bool
    version: str


# LIFECYCLE EVENTS

@app.on_event("startup")
async def startup_event():
    """Load model on startup with comprehensive error handling."""
    global predictor
    
    logger.info("=" * 70)
    logger.info("Starting IVF Patient Response Prediction API")
    logger.info("=" * 70)
    
    try:
        logger.info("Loading model...")
        predictor = IVFPredictor()
        logger.info("✓ Model loaded successfully")
        
        # Test prediction to verify model works
        logger.info("Testing model functionality...")
        test_data = {
            'age': 30,
            'amh': 2.5,
            'afc': 10,
            'n_follicles': 10,
            'e2_day5': 500,
            'cycle_number': 1,
            'protocol': 'flexible antagonist'
        }
        _ = predictor.predict(test_data, include_explanation=False)
        logger.info("✓ Model test prediction successful")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.critical("=" * 70)
        logger.critical("✗ MODEL FILES NOT FOUND")
        logger.critical("=" * 70)
        logger.critical(f"Error: {str(e)}")
        logger.critical("Please train the model first:")
        logger.critical("  python src/model/train.py")
        logger.critical("=" * 70)
        predictor = None
        # Don't raise - let server start but return unhealthy status
        
    except Exception as e:
        logger.critical("=" * 70)
        logger.critical("✗ MODEL LOADING FAILED")
        logger.critical("=" * 70)
        logger.critical(f"Error: {str(e)}")
        logger.critical("=" * 70)
        predictor = None
        raise  # Critical error - prevent server start


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("=" * 70)
    logger.info("Shutting down API")
    logger.info("=" * 70)


# API ENDPOINTS

@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "IVF Patient Response Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "batch": "/predict/batch (POST)",
            "model_info": "/model/info",
            "protocols": "/protocols"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and model status"
)
async def health_check():
    """
    Comprehensive health check endpoint.
    Tests both model loading and functionality.
    """
    model_loaded = predictor is not None
    model_functional = False
    
    if model_loaded:
        try:
            # Test prediction to verify model actually works
            test_data = {
                'age': 30,
                'amh': 2.5,
                'afc': 10,
                'n_follicles': 10,
                'e2_day5': 500,
                'cycle_number': 1,
                'protocol': 'flexible antagonist'
            }
            _ = predictor.predict(test_data, include_explanation=False)
            model_functional = True
        except Exception as e:
            logger.error(f"Health check prediction failed: {str(e)}")
    
    status_str = "healthy" if (model_loaded and model_functional) else "unhealthy"
    
    return {
        "status": status_str,
        "model_loaded": model_loaded,
        "model_functional": model_functional,
        "version": "1.0.0"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make prediction",
    description="Predict IVF patient response based on clinical parameters",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_class": 2,
                        "predicted_label": "high",
                        "confidence": "72.3%",
                        "probabilities": {
                            "low": "8.5%",
                            "optimal": "19.2%",
                            "high": "72.3%"
                        },
                        "interpretation": "72% chance this patient is high responsive",
                        "clinical_recommendation": "⚠️ HIGH RESPONSE PREDICTED..."
                    }
                }
            }
        },
        400: {"description": "Invalid input data"},
        503: {"description": "Model not loaded"}
    }
)
async def predict(patient: PatientData):
    """
    Make prediction for a single patient.
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        # Convert Pydantic model to dict
        patient_data = patient.dict()
        
        # Make prediction
        result = predictor.predict(patient_data, include_explanation=True)
        
        # Log prediction for monitoring
        logger.info(
            f"Prediction: {result['predicted_label']} "
            f"({result['confidence']}) | "
            f"Age={patient.age}, AMH={patient.amh}"
        )
        
        return result
        
    except ValueError as e:
        # Input validation error
        logger.warning(f"Invalid input: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        # Unexpected error
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    summary="Batch prediction",
    description="Make predictions for multiple patients",
    response_model=List[PredictionResponse]
)
async def predict_batch(patients: List[PatientData]):
    """
    Make predictions for multiple patients.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if len(patients) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 patients per batch request"
        )
    
    try:
        results = []
        
        for idx, patient in enumerate(patients):
            patient_data = patient.dict()
            result = predictor.predict(patient_data, include_explanation=True)
            results.append(result)
        
        logger.info(f"Batch prediction completed: {len(results)} patients")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/model/info",
    summary="Model information",
    description="Get information about the loaded model"
)
async def model_info():
    """Get model metadata and information."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        return {
            "model_name": predictor.metadata.get('best_model', 'Unknown'),
            "feature_names": predictor.feature_names,
            "class_names": predictor.class_names,
            "num_features": len(predictor.feature_names) if predictor.feature_names else 0,
            "num_classes": len(predictor.class_names) if predictor.class_names else 0,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@app.get(
    "/protocols",
    summary="Get available protocols",
    description="List all available stimulation protocols"
)
async def get_protocols():
    """Get list of available stimulation protocols with descriptions."""
    return {
        "protocols": [
            {
                "name": "flexible antagonist",
                "code": 0,
                "description": "GnRH antagonist started when follicle reaches 14mm",
                "clinical_use": "Most common, flexible timing"
            },
            {
                "name": "fixed antagonist",
                "code": 1,
                "description": "GnRH antagonist started on day 5/6 of stimulation",
                "clinical_use": "Fixed protocol, predictable timing"
            },
            {
                "name": "agonist",
                "code": 2,
                "description": "Long protocol with pituitary suppression",
                "clinical_use": "Traditional approach, longer duration"
            }
        ]
    }


# ERROR HANDLERS

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions (input validation)."""
    logger.warning(f"ValueError: {str(exc)}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error. Please contact administrator."
    )


# MAIN ENTRY POINT

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("  IVF PATIENT RESPONSE PREDICTION API")
    print("=" * 70)
    print("\n🚀 Starting server...")
    print(f"📍 Project root: {PROJECT_ROOT}")
    print("\n📚 Documentation:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc:      http://localhost:8000/redoc")
    print("\n🔗 Endpoints:")
    print("   - Health:     http://localhost:8000/health")
    print("   - Predict:    http://localhost:8000/predict (POST)")
    print("\n⚠️  Press CTRL+C to stop\n")
    print("=" * 70 + "\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")