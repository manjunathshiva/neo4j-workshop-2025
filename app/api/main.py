"""
FastAPI backend for Knowledge Graph RAG System.
Provides REST API endpoints and health checks.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from connections import get_database_manager
from startup_validation import StartupValidator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Graph RAG API",
    description="REST API for Knowledge Graph RAG System with Neo4j and Qdrant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_startup_complete = False
_startup_results = None
_config = None
_db_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global _startup_complete, _startup_results, _config, _db_manager
    
    logger.info("Starting Knowledge Graph RAG API...")
    
    try:
        # Load configuration
        _config = get_config()
        _db_manager = get_database_manager()
        
        # Run startup validation
        validator = StartupValidator()
        _startup_results = await validator.validate_all(quick_mode=False)
        
        # Check if system is ready
        if _startup_results["overall_status"] in ["healthy", "degraded"]:
            _startup_complete = True
            logger.info("API startup completed successfully")
        else:
            logger.error("API startup validation failed")
            _startup_complete = False
            
    except Exception as e:
        logger.error(f"Error during API startup: {str(e)}")
        _startup_complete = False
        _startup_results = {
            "overall_status": "critical_errors",
            "critical_errors": [str(e)]
        }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    global _db_manager
    
    logger.info("Shutting down Knowledge Graph RAG API...")
    
    if _db_manager:
        try:
            await _db_manager.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")


# Health check models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    environment: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""
    status: str
    timestamp: str
    version: str
    environment: str
    components: Dict[str, Any]
    startup_complete: bool


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    ready: bool
    status: str
    message: str
    components: Dict[str, bool]


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Knowledge Graph RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if the API is running.
    """
    return HealthResponse(
        status="healthy" if _startup_complete else "starting",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        environment=_config.app.environment if _config else "unknown"
    )


@app.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check endpoint.
    Returns comprehensive system status including all components.
    """
    if not _startup_results:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System startup not yet complete"
        )
    
    # Extract component status
    components = {
        "configuration": {
            "status": "healthy" if _startup_results.get("configuration", {}).get("valid", False) else "unhealthy",
            "details": _startup_results.get("configuration", {})
        },
        "neo4j": {
            "status": "healthy" if _startup_results.get("database_connections", {}).get("neo4j", {}).get("connected", False) else "unhealthy",
            "message": _startup_results.get("database_connections", {}).get("neo4j", {}).get("message", "Unknown")
        },
        "qdrant": {
            "status": "healthy" if _startup_results.get("database_connections", {}).get("qdrant", {}).get("connected", False) else "unhealthy",
            "message": _startup_results.get("database_connections", {}).get("qdrant", {}).get("message", "Unknown")
        }
    }
    
    overall_status = _startup_results.get("overall_status", "unknown")
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        environment=_config.app.environment if _config else "unknown",
        components=components,
        startup_complete=_startup_complete
    )


@app.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check endpoint.
    Returns 200 if the system is ready to accept requests.
    Returns 503 if the system is not ready.
    """
    if not _startup_complete or not _startup_results:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not ready"
        )
    
    overall_status = _startup_results.get("overall_status", "unknown")
    
    # Check component readiness
    neo4j_ready = _startup_results.get("database_connections", {}).get("neo4j", {}).get("connected", False)
    qdrant_ready = _startup_results.get("database_connections", {}).get("qdrant", {}).get("connected", False)
    config_ready = _startup_results.get("configuration", {}).get("valid", False)
    
    components = {
        "configuration": config_ready,
        "neo4j": neo4j_ready,
        "qdrant": qdrant_ready
    }
    
    # System is ready if configuration is valid and at least one database is connected
    is_ready = config_ready and (neo4j_ready or qdrant_ready)
    
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not ready - check component status"
        )
    
    return ReadinessResponse(
        ready=is_ready,
        status=overall_status,
        message="System is ready to accept requests",
        components=components
    )


@app.get("/health/live", response_model=Dict[str, str])
async def liveness_check():
    """
    Liveness check endpoint.
    Returns 200 if the API process is alive.
    This is a simple check that doesn't validate dependencies.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health/startup", response_model=Dict[str, Any])
async def startup_status():
    """
    Startup status endpoint.
    Returns the complete startup validation results.
    """
    if not _startup_results:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Startup validation not yet complete"
        )
    
    return _startup_results


@app.post("/health/validate", response_model=Dict[str, Any])
async def trigger_validation():
    """
    Trigger a new validation check.
    Useful for re-checking system status after configuration changes.
    """
    global _startup_results
    
    try:
        validator = StartupValidator()
        _startup_results = await validator.validate_all(quick_mode=False)
        
        return {
            "message": "Validation completed",
            "results": _startup_results
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@app.get("/config/info", response_model=Dict[str, Any])
async def config_info():
    """
    Get configuration information (non-sensitive).
    Returns configuration summary without exposing credentials.
    """
    if not _config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    return {
        "environment": _config.app.environment,
        "log_level": _config.app.log_level,
        "demo_mode": _config.app.demo_mode,
        "ports": {
            "streamlit": _config.app.streamlit_port,
            "fastapi": _config.app.fastapi_port
        },
        "database_info": _config.get_database_connection_info(),
        "llm_info": _config.get_llm_info()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    config = get_config()
    
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.app.fastapi_port,
        reload=config.app.environment == "development",
        log_level=config.app.log_level.lower()
    )
