from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up additional logging handlers
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=False,  # Set to False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse('static/index.html')

# Handle favicon request to avoid 404
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)  # No content, suppresses 404

# Define input models
class InvestorInput(BaseModel):
    type: str
    location: str
    avg_check_size: float
    min_roi: float
    risk_appetite: float
    years_active: float
    total_investments: float
    preferred_sectors: List[str]
    preferred_stages: List[str]
    thesis: str

class StartupInput(BaseModel):
    sector: str
    stage: str
    location: str
    founding_date: str
    employees: int
    mrr: float
    growth_rate: float
    burn_rate: float
    funding_to_date: float
    description: str
    last_valuation: float

# Load pre-trained models and vectorizers
try:
    compat_model = joblib.load('compatibility_model.joblib')
    traction_model = joblib.load('traction_model.joblib')
    industry_model = joblib.load('industry_model.joblib')
    thesis_vectorizer = joblib.load('thesis_vectorizer.joblib')
    description_vectorizer = joblib.load('description_vectorizer.joblib')
    logger.info("Models and vectorizers loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models or vectorizers: {str(e)}")
    raise

# Define global constants
SECTORS = ['Tech', 'Healthcare', 'Fintech', 'Consumer', 'Enterprise', 'AI/ML', 'CleanTech']
STAGES = ['Pre-seed', 'Seed', 'Series A', 'Series B', 'Growth']
INVESTOR_TYPES = {"VC": 0, "Angel": 1, "Corporate": 2, "PE": 3}

# Preprocessing function (improved version)
def preprocess_input(investor_data, startup_data):
    try:
        if investor_data:
            # Numeric features
            investor_numeric = np.array([
                investor_data.avg_check_size,
                investor_data.min_roi,
                investor_data.risk_appetite,
                investor_data.years_active,
                investor_data.total_investments
            ], dtype=np.float32)

            # Transform thesis text
            thesis_vec = thesis_vectorizer.transform([investor_data.thesis]).toarray()
            if thesis_vec.shape[1] != 50:  # Adjust based on your TF-IDF vectorizer's max_features
                logger.warning(f"Unexpected thesis vector shape: {thesis_vec.shape}")
                thesis_vec = np.pad(thesis_vec, ((0, 0), (0, 50 - thesis_vec.shape[1])), 'constant')

            # One-hot encoding for type
            investor_type_vec = np.zeros(4, dtype=np.float32)
            if investor_data.type.upper() in INVESTOR_TYPES:
                investor_type_vec[INVESTOR_TYPES[investor_data.type.upper()]] = 1

            # Sector and stage preferences (fixed size vectors)
            sector_vec = np.zeros(len(SECTORS), dtype=np.float32)
            stage_vec = np.zeros(len(STAGES), dtype=np.float32)

            for sector in investor_data.preferred_sectors:
                if sector in SECTORS:
                    sector_vec[SECTORS.index(sector)] = 1
            for stage in investor_data.preferred_stages:
                if stage in STAGES:
                    stage_vec[STAGES.index(stage)] = 1

            # Combine features
            investor_vec = np.concatenate([
                investor_numeric,
                investor_type_vec,
                sector_vec,
                stage_vec,
                thesis_vec[0]
            ], dtype=np.float32)
            
            # Pad investor vector if needed
            if investor_vec.shape[0] < 250:  # Half of expected 489 features
                investor_vec = np.pad(investor_vec, (0, 250 - investor_vec.shape[0]), 'constant')
        else:
            investor_vec = np.zeros(250, dtype=np.float32)  # Placeholder for investor features

        if startup_data:
            # Numeric features
            startup_numeric = np.array([
                startup_data.employees,
                startup_data.mrr,
                startup_data.growth_rate,
                startup_data.burn_rate,
                startup_data.funding_to_date,
                startup_data.last_valuation
            ], dtype=np.float32)

            # Transform description text
            desc_vec = description_vectorizer.transform([startup_data.description]).toarray()
            if desc_vec.shape[1] != 50:  # Adjust based on your TF-IDF vectorizer's max_features
                logger.warning(f"Unexpected description vector shape: {desc_vec.shape}")
                desc_vec = np.pad(desc_vec, ((0, 0), (0, 50 - desc_vec.shape[1])), 'constant')

            # Sector and stage encoding
            sector_vec = np.zeros(7, dtype=np.float32)  # Match investor sectors
            stage_vec = np.zeros(5, dtype=np.float32)   # Match investor stages
            if startup_data.sector in SECTORS:
                sector_vec[SECTORS.index(startup_data.sector)] = 1
            if startup_data.stage in STAGES:
                stage_vec[STAGES.index(startup_data.stage)] = 1

            # Create base startup vector
            base_startup_vec = np.concatenate([
                startup_numeric,
                sector_vec,
                stage_vec,
                desc_vec[0]
            ], dtype=np.float32)

            # Create two versions of the startup vector with different paddings
            # One for compatibility model (239 features)
            startup_vec_compat = base_startup_vec.copy()
            if startup_vec_compat.shape[0] < 239:
                startup_vec_compat = np.pad(startup_vec_compat, (0, 239 - startup_vec_compat.shape[0]), 'constant')
            
            # One for traction model (308 features)
            startup_vec_traction = base_startup_vec.copy()
            if startup_vec_traction.shape[0] < 308:
                startup_vec_traction = np.pad(startup_vec_traction, (0, 308 - startup_vec_traction.shape[0]), 'constant')
            
            # Store both vectors
            startup_vectors = (startup_vec_compat, startup_vec_traction)
        else:
            startup_vectors = (np.zeros(239, dtype=np.float32), np.zeros(308, dtype=np.float32))

        logger.info(f"Investor vector shape: {investor_vec.shape}, Startup vectors shapes: compat={startup_vectors[0].shape}, traction={startup_vectors[1].shape}")
        return investor_vec, startup_vectors
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.post("/predict_compatibility/")
async def predict_compatibility(investor: InvestorInput, startup: StartupInput):
    try:
        logger.info(f"Processing compatibility prediction for investor type: {investor.type} and startup sector: {startup.sector}")
        
        # Get feature vectors - now returns (compat_vec, traction_vec) for startup
        investor_vec, (startup_vec_compat, _) = preprocess_input(investor, startup)
        logger.info(f"Generated vectors - Investor: {investor_vec.shape}, Startup: {startup_vec_compat.shape}")
        
        # Combine vectors for compatibility prediction
        combined_vec = np.concatenate([investor_vec, startup_vec_compat])
        logger.info(f"Combined vector shape: {combined_vec.shape}")
        
        # Check for NaN values
        if np.isnan(combined_vec).any():
            logger.error("NaN values detected in feature vector")
            return {"error": "Invalid input values detected", "compatibility_score": 0.0}
            
        # Reshape and predict
        reshaped_vec = combined_vec.reshape(1, -1)
        logger.info(f"Making prediction with vector shape: {reshaped_vec.shape}")
        
        # Get model prediction
        compatibility_score = float(compat_model.predict_proba(reshaped_vec)[0][1])
        logger.info(f"Prediction complete. Score: {compatibility_score}")
        
        if np.isnan(compatibility_score):
            logger.error("Model produced NaN score")
            return {"error": "Model produced invalid score", "compatibility_score": 0.0}
            
        return {
            "compatibility_score": compatibility_score,
            "investor_features": investor_vec.shape[0],
            "startup_features": startup_vec_compat.shape[0],
            "total_features": combined_vec.shape[0]
        }
    except Exception as e:
        logger.error(f"Error in predict_compatibility: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "compatibility_score": 0.0}

@app.post("/predict_traction/")
async def predict_traction(startup: StartupInput):
    try:
        logger.info(f"Processing traction prediction for startup: {startup.sector}")
        
        # Get feature vector - use the traction-specific vector
        _, (_, startup_vec_traction) = preprocess_input(None, startup)
        logger.info(f"Generated startup vector shape: {startup_vec_traction.shape}")
        
        # Check for NaN values
        if np.isnan(startup_vec_traction).any():
            logger.error("NaN values detected in feature vector")
            return {"error": "Invalid input values detected", "traction_score": 0.0}
        
        # Reshape and predict
        reshaped_vec = startup_vec_traction.reshape(1, -1)
        logger.info(f"Making prediction with vector shape: {reshaped_vec.shape}")
        
        # Get model prediction
        traction_score = float(traction_model.predict(reshaped_vec)[0])
        logger.info(f"Prediction complete. Score: {traction_score}")
        
        if np.isnan(traction_score):
            logger.error("Model produced NaN score")
            return {"error": "Model produced invalid score", "traction_score": 0.0}
            
        return {
            "traction_score": traction_score,
            "features_used": startup_vec_traction.shape[0]
        }
    except Exception as e:
        logger.error(f"Error in predict_traction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "traction_score": 0.0}

@app.get("/sector_similarity/")
async def sector_similarity(sector1: str, sector2: str):
    try:
        logger.info(f"Processing sector similarity between {sector1} and {sector2}")
        
        # Validate sectors
        if sector1 not in SECTORS or sector2 not in SECTORS:
            invalid_sectors = [s for s in [sector1, sector2] if s not in SECTORS]
            return {"error": f"Invalid sector(s): {', '.join(invalid_sectors)}", "similarity_score": 0.0}
            
        # Get similarity score
        similarity = industry_model.get_sector_similarity(sector1, sector2)
        similarity_float = float(similarity)
        
        if np.isnan(similarity_float):
            logger.error("Model produced NaN similarity score")
            return {"error": "Invalid similarity score", "similarity_score": 0.0}
            
        logger.info(f"Similarity calculation complete. Score: {similarity_float}")
        return {"similarity_score": similarity_float}
        
    except Exception as e:
        logger.error(f"Error in sector_similarity: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "similarity_score": 0.0}