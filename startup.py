#!/usr/bin/env python3
"""
Startup script for Knowledge Graph RAG System.
Validates configuration and database connections before launching the application.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.config import get_config
from app.connections import get_database_manager

logger = logging.getLogger(__name__)

async def validate_startup():
    """
    Validate system configuration and database connections.
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    print("🚀 Starting Knowledge Graph RAG System...")
    print("=" * 60)
    
    try:
        # Load and validate configuration
        print("📋 Validating configuration...")
        config = get_config()
        validation_results = config.validate_configuration()
        
        if validation_results["errors"]:
            print("❌ Configuration errors found:")
            for error in validation_results["errors"]:
                print(f"   • {error}")
            print("\n💡 Tip: Check your .env file and ensure all required variables are set")
            return False
        
        if validation_results["warnings"]:
            print("⚠️  Configuration warnings:")
            for warning in validation_results["warnings"]:
                print(f"   • {warning}")
        
        print("✅ Configuration validation passed")
        
        # Display configuration summary
        print("\n📊 Configuration Summary:")
        db_info = config.get_database_connection_info()
        llm_info = config.get_llm_info()
        
        print(f"   • Neo4j URI: {db_info['neo4j_uri']}")
        print(f"   • Qdrant URL: {db_info['qdrant_url']}")
        print(f"   • Primary LLM: {llm_info['primary_llm']}")
        print(f"   • Embedding Model: {llm_info['embedding_model']}")
        print(f"   • Environment: {config.app.environment}")
        
        # Test database connections
        print("\n🔗 Testing database connections...")
        db_manager = get_database_manager()
        connection_status = await db_manager.initialize()
        
        neo4j_connected = connection_status["neo4j"]["connected"]
        qdrant_connected = connection_status["qdrant"]["connected"]
        
        if neo4j_connected:
            print(f"✅ Neo4j: {connection_status['neo4j']['message']}")
        else:
            print(f"❌ Neo4j: {connection_status['neo4j']['message']}")
        
        if qdrant_connected:
            print(f"✅ Qdrant: {connection_status['qdrant']['message']}")
        else:
            print(f"❌ Qdrant: {connection_status['qdrant']['message']}")
        
        # Graceful degradation - allow system to start if at least one database is connected
        if not connection_status["overall_success"]:
            if not neo4j_connected and not qdrant_connected:
                print("\n❌ Both database connections failed!")
                print("The system cannot operate without at least one database connection.")
                print("\n💡 Troubleshooting tips:")
                print("   • Verify your .env file has correct credentials")
                print("   • Check that your cloud databases are running")
                print("   • Ensure network connectivity to cloud services")
                return False
            else:
                print("\n⚠️  Partial database connectivity - system will run with limited functionality")
                if not neo4j_connected:
                    print("   • Graph RAG features will be unavailable")
                if not qdrant_connected:
                    print("   • Vector search and Hybrid RAG will be unavailable")
        else:
            print("\n✅ All database connections validated successfully!")
        
        # Display next steps
        print("\n🎯 System Ready!")
        print("=" * 60)
        print("Next steps:")
        print("1. Start Streamlit app: streamlit run app/main.py --server.port 8501")
        print("2. Or start FastAPI backend: uvicorn app.api.main:app --host 0.0.0.0 --port 8000")
        print("3. Or use the convenience script: bash .devcontainer/start-services.sh")
        print("4. Access the web interface via Codespace port forwarding")
        print("\nFor workshop participants:")
        print("• Upload documents through the web interface")
        print("• Try both Graph RAG and Hybrid RAG approaches")
        print("• Explore the knowledge graph visualization")
        print("\nHealth check endpoints (when FastAPI is running):")
        print("• http://localhost:8000/health - Basic health check")
        print("• http://localhost:8000/health/detailed - Detailed component status")
        print("• http://localhost:8000/health/ready - Readiness probe")
        print("• http://localhost:8000/docs - API documentation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {str(e)}")
        logger.exception("Validation error")
        return False

def main():
    """Main startup function."""
    try:
        success = asyncio.run(validate_startup())
        if not success:
            print("\n💥 Startup validation failed. Please fix the issues above.")
            sys.exit(1)
        
        print("\n🎉 Startup validation completed successfully!")
        print("You can now launch the application components.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error during startup: {str(e)}")
        logger.exception("Startup error")
        sys.exit(1)

if __name__ == "__main__":
    main()