"""
Startup validation for Knowledge Graph RAG System.
Validates all cloud database connections and configuration on system startup.
"""

import logging
import asyncio
import sys
from typing import Dict, Any, List
from datetime import datetime

from config import get_config
from connections import get_database_manager

logger = logging.getLogger(__name__)

class StartupValidator:
    """Handles comprehensive startup validation for all system components."""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_database_manager()
        self.validation_results: Dict[str, Any] = {}
        self.critical_errors: List[str] = []
        self.warnings: List[str] = []
    
    async def validate_all(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive startup validation.
        
        Args:
            quick_mode: If True, perform faster validation without full connection setup
            
        Returns:
            Dictionary with complete validation results
        """
        logger.info("Starting comprehensive system validation...")
        start_time = datetime.now()
        
        self.validation_results = {
            "timestamp": start_time.isoformat(),
            "quick_mode": quick_mode,
            "configuration": {},
            "database_connections": {},
            "overall_status": "unknown",
            "critical_errors": [],
            "warnings": [],
            "recommendations": [],
            "validation_time_seconds": 0
        }
        
        # Step 1: Validate configuration
        logger.info("Validating configuration...")
        config_validation = self._validate_configuration()
        self.validation_results["configuration"] = config_validation
        
        # Step 2: Validate database connections
        logger.info("Validating database connections...")
        db_validation = await self._validate_database_connections(quick_mode)
        self.validation_results["database_connections"] = db_validation
        
        # Step 3: Generate overall status and recommendations
        self._generate_overall_status()
        self._generate_recommendations()
        
        # Calculate validation time
        end_time = datetime.now()
        self.validation_results["validation_time_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"Validation completed in {self.validation_results['validation_time_seconds']:.2f} seconds")
        return self.validation_results
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration."""
        try:
            config_results = self.config.validate_configuration()
            
            # Add additional configuration checks
            config_results["environment_info"] = {
                "environment": self.config.app.environment,
                "log_level": self.config.app.log_level,
                "demo_mode": self.config.app.demo_mode,
                "ports": {
                    "streamlit": self.config.app.streamlit_port,
                    "fastapi": self.config.app.fastapi_port
                }
            }
            
            # Check for critical configuration issues
            if not config_results["valid"]:
                self.critical_errors.extend(config_results["errors"])
            
            if config_results["warnings"]:
                self.warnings.extend(config_results["warnings"])
            
            return config_results
            
        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            logger.error(error_msg)
            self.critical_errors.append(error_msg)
            return {
                "valid": False,
                "errors": [error_msg],
                "warnings": [],
                "database_status": {},
                "llm_status": {}
            }
    
    async def _validate_database_connections(self, quick_mode: bool) -> Dict[str, Any]:
        """Validate database connections."""
        try:
            # Use the database manager's initialization method
            db_results = await self.db_manager.initialize(validate_only=quick_mode)
            
            # Add connection errors to critical errors if both databases fail
            if not db_results["overall_success"]:
                neo4j_failed = not db_results["neo4j"]["connected"]
                qdrant_failed = not db_results["qdrant"]["connected"]
                
                if neo4j_failed and qdrant_failed:
                    self.critical_errors.append("Both Neo4j and Qdrant connections failed - system cannot operate")
                elif neo4j_failed:
                    self.warnings.append("Neo4j connection failed - graph operations will not be available")
                elif qdrant_failed:
                    self.warnings.append("Qdrant connection failed - vector search will not be available")
            
            # Add any database-specific errors
            if db_results["errors"]:
                self.critical_errors.extend(db_results["errors"])
            
            if db_results["warnings"]:
                self.warnings.extend(db_results["warnings"])
            
            return db_results
            
        except Exception as e:
            error_msg = f"Database validation failed: {str(e)}"
            logger.error(error_msg)
            self.critical_errors.append(error_msg)
            return {
                "neo4j": {"connected": False, "message": error_msg},
                "qdrant": {"connected": False, "message": error_msg},
                "overall_success": False,
                "errors": [error_msg]
            }
    
    def _generate_overall_status(self):
        """Generate overall system status based on validation results."""
        config_valid = self.validation_results["configuration"].get("valid", False)
        db_success = self.validation_results["database_connections"].get("overall_success", False)
        
        if self.critical_errors:
            self.validation_results["overall_status"] = "critical_errors"
        elif config_valid and db_success:
            self.validation_results["overall_status"] = "healthy"
        elif config_valid and not db_success:
            self.validation_results["overall_status"] = "degraded"
        else:
            self.validation_results["overall_status"] = "configuration_errors"
        
        self.validation_results["critical_errors"] = self.critical_errors
        self.validation_results["warnings"] = self.warnings
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Configuration recommendations
        config_results = self.validation_results["configuration"]
        if not config_results.get("valid", False):
            recommendations.append("Fix configuration errors in .env file before proceeding")
        
        # Database recommendations
        db_results = self.validation_results["database_connections"]
        if not db_results.get("overall_success", False):
            neo4j_connected = db_results.get("neo4j", {}).get("connected", False)
            qdrant_connected = db_results.get("qdrant", {}).get("connected", False)
            
            if not neo4j_connected:
                neo4j_error_type = db_results.get("neo4j", {}).get("error_type")
                if neo4j_error_type == "AuthError":
                    recommendations.append("Check Neo4j username and password in .env file")
                elif neo4j_error_type == "ServiceUnavailable":
                    recommendations.append("Verify Neo4j URI and ensure the service is running")
                else:
                    recommendations.append("Check Neo4j connection settings and network connectivity")
            
            if not qdrant_connected:
                qdrant_error_type = db_results.get("qdrant", {}).get("error_type")
                if qdrant_error_type == "AuthenticationError":
                    recommendations.append("Check Qdrant API key in .env file")
                elif qdrant_error_type == "NotFoundError":
                    recommendations.append("Verify Qdrant URL and ensure the service is accessible")
                else:
                    recommendations.append("Check Qdrant connection settings and network connectivity")
        
        # LLM recommendations
        llm_status = config_results.get("llm_status", {})
        primary_llm = llm_status.get("primary_llm", "groq")
        if primary_llm == "groq" and not llm_status.get("groq_configured", False):
            recommendations.append("Configure Groq API key for LLM functionality")
        elif primary_llm == "openai" and not llm_status.get("openai_configured", False):
            recommendations.append("Configure OpenAI API key for LLM functionality")
        
        # General recommendations
        if self.validation_results["overall_status"] == "healthy":
            recommendations.append("System is ready for operation")
        elif self.validation_results["overall_status"] == "degraded":
            recommendations.append("System can operate with limited functionality")
        
        self.validation_results["recommendations"] = recommendations
    
    def print_validation_summary(self):
        """Print a formatted validation summary to console."""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH RAG SYSTEM - STARTUP VALIDATION")
        print("="*60)
        
        # Overall status
        status = self.validation_results.get("overall_status", "unknown")
        status_symbols = {
            "healthy": "✅",
            "degraded": "⚠️",
            "configuration_errors": "❌",
            "critical_errors": "❌",
            "unknown": "❓"
        }
        
        print(f"\nOverall Status: {status_symbols.get(status, '❓')} {status.upper()}")
        
        # Configuration status
        config_valid = self.validation_results.get("configuration", {}).get("valid", False)
        print(f"Configuration: {'✅ Valid' if config_valid else '❌ Invalid'}")
        
        # Database status
        db_results = self.validation_results.get("database_connections", {})
        neo4j_connected = db_results.get("neo4j", {}).get("connected", False)
        qdrant_connected = db_results.get("qdrant", {}).get("connected", False)
        
        print(f"Neo4j Connection: {'✅ Connected' if neo4j_connected else '❌ Failed'}")
        print(f"Qdrant Connection: {'✅ Connected' if qdrant_connected else '❌ Failed'}")
        
        # Errors and warnings
        if self.critical_errors:
            print(f"\n❌ Critical Errors ({len(self.critical_errors)}):")
            for error in self.critical_errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Recommendations
        recommendations = self.validation_results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        validation_time = self.validation_results.get("validation_time_seconds", 0)
        print(f"\nValidation completed in {validation_time:.2f} seconds")
        print("="*60 + "\n")

async def validate_startup(quick_mode: bool = False, print_summary: bool = True) -> Dict[str, Any]:
    """
    Main function to validate system startup.
    
    Args:
        quick_mode: If True, perform faster validation without full connection setup
        print_summary: If True, print validation summary to console
        
    Returns:
        Dictionary with validation results
    """
    validator = StartupValidator()
    results = await validator.validate_all(quick_mode=quick_mode)
    
    if print_summary:
        validator.print_validation_summary()
    
    return results

def validate_startup_sync(quick_mode: bool = False, print_summary: bool = True) -> Dict[str, Any]:
    """
    Synchronous wrapper for startup validation.
    
    Args:
        quick_mode: If True, perform faster validation without full connection setup
        print_summary: If True, print validation summary to console
        
    Returns:
        Dictionary with validation results
    """
    return asyncio.run(validate_startup(quick_mode=quick_mode, print_summary=print_summary))

if __name__ == "__main__":
    # Command line interface for startup validation
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Knowledge Graph RAG System startup")
    parser.add_argument("--quick", action="store_true", help="Perform quick validation without full connection setup")
    parser.add_argument("--quiet", action="store_true", help="Don't print validation summary")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    try:
        results = validate_startup_sync(
            quick_mode=args.quick,
            print_summary=not args.quiet and not args.json
        )
        
        if args.json:
            import json
            print(json.dumps(results, indent=2, default=str))
        
        # Exit with appropriate code
        if results["overall_status"] in ["critical_errors", "configuration_errors"]:
            sys.exit(1)
        elif results["overall_status"] == "degraded":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Startup validation failed: {str(e)}")
        if not args.quiet:
            print(f"❌ Startup validation failed: {str(e)}")
        sys.exit(1)