"""
Configuration management for Knowledge Graph RAG System.
Handles environment variables, database connections, and application settings.
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    
    # Neo4j Configuration
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    
    # Qdrant Configuration
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    
    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, v):
        if not v.startswith(("neo4j://", "neo4j+s://", "bolt://", "bolt+s://")):
            raise ValueError("Neo4j URI must start with neo4j://, neo4j+s://, bolt://, or bolt+s://")
        return v
    
    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v

class LLMConfig(BaseSettings):
    """LLM and API configuration."""
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_token: Optional[str] = Field(default=None, env="HUGGINGFACE_API_TOKEN")
    
    # Model Configuration
    primary_llm: str = Field(default="groq", env="PRIMARY_LLM")
    text_to_cypher_model: str = Field(default="moonshotai/kimi-k2-instruct-0905", env="TEXT_TO_CYPHER_MODEL")
    embedding_model: str = Field(default="local", env="EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="LOCAL_EMBEDDING_MODEL")
    
    @field_validator("primary_llm")
    @classmethod
    def validate_primary_llm(cls, v):
        allowed_providers = ["groq", "openai", "anthropic", "huggingface"]
        if v not in allowed_providers:
            raise ValueError(f"Primary LLM must be one of: {allowed_providers}")
        return v
    
    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, v):
        allowed_models = ["local", "openai"]
        if v not in allowed_models:
            raise ValueError(f"Embedding model must be one of: {allowed_models}")
        return v

class AppConfig(BaseSettings):
    """Application configuration."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Ports
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    fastapi_port: int = Field(default=8000, env="FASTAPI_PORT")
    
    # Processing Settings
    max_document_size_mb: int = Field(default=10, env="MAX_DOCUMENT_SIZE_MB")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_entities_per_document: int = Field(default=100, env="MAX_ENTITIES_PER_DOCUMENT")
    
    # Workshop Settings
    demo_mode: bool = Field(default=True, env="DEMO_MODE")
    sample_docs_dir: str = Field(default="data/samples", env="SAMPLE_DOCS_DIR")
    show_processing_status: bool = Field(default=True, env="SHOW_PROCESSING_STATUS")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()

class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.app = AppConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.app.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the complete configuration and return status.
        
        Returns:
            Dict containing validation results and any errors.
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "database_status": {},
            "llm_status": {}
        }
        
        # Check database configuration
        try:
            # Validate required database fields
            if not self.database.neo4j_password or self.database.neo4j_password == "your-neo4j-password":
                validation_results["errors"].append("Neo4j password not configured")
                validation_results["valid"] = False
            
            if not self.database.qdrant_api_key or self.database.qdrant_api_key == "your-qdrant-api-key":
                validation_results["errors"].append("Qdrant API key not configured")
                validation_results["valid"] = False
                
            validation_results["database_status"] = {
                "neo4j_configured": bool(self.database.neo4j_password and self.database.neo4j_password != "your-neo4j-password"),
                "qdrant_configured": bool(self.database.qdrant_api_key and self.database.qdrant_api_key != "your-qdrant-api-key")
            }
            
        except Exception as e:
            validation_results["errors"].append(f"Database configuration error: {str(e)}")
            validation_results["valid"] = False
        
        # Check LLM configuration
        try:
            llm_configured = False
            
            if self.llm.primary_llm == "groq" and self.llm.groq_api_key and self.llm.groq_api_key != "your-groq-api-key":
                llm_configured = True
            elif self.llm.primary_llm == "openai" and self.llm.openai_api_key and self.llm.openai_api_key != "your-openai-api-key":
                llm_configured = True
            elif self.llm.primary_llm == "anthropic" and self.llm.anthropic_api_key and self.llm.anthropic_api_key != "your-anthropic-api-key":
                llm_configured = True
            
            if not llm_configured:
                validation_results["warnings"].append(f"Primary LLM ({self.llm.primary_llm}) API key not configured")
            
            validation_results["llm_status"] = {
                "primary_llm": self.llm.primary_llm,
                "groq_configured": bool(self.llm.groq_api_key and self.llm.groq_api_key != "your-groq-api-key"),
                "openai_configured": bool(self.llm.openai_api_key and self.llm.openai_api_key != "your-openai-api-key"),
                "anthropic_configured": bool(self.llm.anthropic_api_key and self.llm.anthropic_api_key != "your-anthropic-api-key"),
                "embedding_model": self.llm.embedding_model
            }
            
        except Exception as e:
            validation_results["errors"].append(f"LLM configuration error: {str(e)}")
            validation_results["valid"] = False
        
        return validation_results
    
    def get_database_connection_info(self) -> Dict[str, str]:
        """Get database connection information for display purposes."""
        return {
            "neo4j_uri": self.database.neo4j_uri,
            "neo4j_username": self.database.neo4j_username,
            "qdrant_url": self.database.qdrant_url,
            "embedding_model": self.llm.embedding_model
        }
    
    def get_llm_info(self) -> Dict[str, str]:
        """Get LLM configuration information for display purposes."""
        return {
            "primary_llm": self.llm.primary_llm,
            "text_to_cypher_model": self.llm.text_to_cypher_model,
            "embedding_model": self.llm.embedding_model,
            "local_embedding_model": self.llm.local_embedding_model
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config