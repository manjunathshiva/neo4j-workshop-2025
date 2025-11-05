"""
Database connection management for Neo4j and Qdrant cloud instances.
Handles connection validation, retry logic, and connection pooling.
"""

import logging
import asyncio
import time
from typing import Optional, Dict, Any, Tuple, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

# Database clients
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    from .config import get_config
except ImportError:
    from config import get_config

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    VALIDATING = "validating"

@dataclass
class ConnectionResult:
    """Result of a connection attempt."""
    success: bool
    message: str
    error_type: Optional[str] = None
    retry_after: Optional[int] = None
    connection_info: Optional[Dict[str, Any]] = None

class Neo4jConnection:
    """Manages Neo4j cloud database connection with enhanced validation and error handling."""
    
    def __init__(self):
        self.config = get_config()
        self._driver: Optional[Driver] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._last_error: Optional[str] = None
        self._connection_info: Dict[str, Any] = {}
        self._retry_count = 0
        self._max_retries = 3
        self._retry_delay = 2  # seconds
    
    def connect(self, max_retries: int = 3) -> Driver:
        """
        Establish connection to Neo4j cloud instance with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            
        Returns:
            Neo4j driver instance
            
        Raises:
            ConnectionError: If connection fails after all retries
        """
        self._status = ConnectionStatus.CONNECTING
        self._retry_count = 0
        
        for attempt in range(max_retries):
            try:
                self._retry_count = attempt + 1
                logger.info(f"Neo4j connection attempt {self._retry_count}/{max_retries}")
                
                self._driver = GraphDatabase.driver(
                    self.config.database.neo4j_uri,
                    auth=(self.config.database.neo4j_username, self.config.database.neo4j_password),
                    connection_timeout=30,
                    max_connection_lifetime=3600
                )
                
                # Test connection with comprehensive validation
                validation_result = self._validate_connection()
                if not validation_result.success:
                    self._driver.close()
                    self._driver = None
                    raise ConnectionError(validation_result.message)
                
                self._status = ConnectionStatus.CONNECTED
                self._last_error = None
                self._connection_info = validation_result.connection_info or {}
                
                logger.info(f"Successfully connected to Neo4j cloud instance on attempt {self._retry_count}")
                return self._driver
                
            except (ServiceUnavailable, AuthError, ConfigurationError) as e:
                self._last_error = str(e)
                error_type = type(e).__name__
                logger.warning(f"Neo4j connection attempt {self._retry_count} failed ({error_type}): {str(e)}")
                
                if attempt < max_retries - 1:
                    sleep_time = self._retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self._status = ConnectionStatus.ERROR
                    raise ConnectionError(f"Neo4j connection failed after {max_retries} attempts: {str(e)}")
                    
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Unexpected error during Neo4j connection: {str(e)}")
                self._status = ConnectionStatus.ERROR
                raise ConnectionError(f"Neo4j connection failed: {str(e)}")
    
    def _validate_connection(self) -> ConnectionResult:
        """
        Validate Neo4j connection with comprehensive checks.
        
        Returns:
            ConnectionResult with validation details
        """
        try:
            with self._driver.session() as session:
                # Test basic connectivity
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value != 1:
                    return ConnectionResult(False, "Basic connectivity test failed")
                
                # Get basic database statistics (compatible with free tier)
                stats = session.run("MATCH (n) RETURN count(n) as node_count").single()
                node_count = stats["node_count"] if stats else 0
                
                # Try to get database name (fallback if not available)
                try:
                    db_result = session.run("CALL db.info() YIELD name")
                    db_record = db_result.single()
                    db_name = db_record["name"] if db_record else "neo4j"
                except:
                    db_name = "neo4j"
                
                connection_info = {
                    "database_name": db_name,
                    "node_count": node_count,
                    "uri": self.config.database.neo4j_uri,
                    "username": self.config.database.neo4j_username
                }
                
                return ConnectionResult(
                    success=True,
                    message="Neo4j connection validated successfully",
                    connection_info=connection_info
                )
                
        except Exception as e:
            return ConnectionResult(False, f"Connection validation failed: {str(e)}")
    
    def test_connection(self) -> ConnectionResult:
        """
        Test Neo4j connection without storing the driver.
        
        Returns:
            ConnectionResult with test details
        """
        try:
            driver = GraphDatabase.driver(
                self.config.database.neo4j_uri,
                auth=(self.config.database.neo4j_username, self.config.database.neo4j_password),
                connection_timeout=10
            )
            
            with driver.session() as session:
                # Basic connectivity test
                result = session.run("RETURN 'Connection successful' as message, datetime() as timestamp")
                record = result.single()
                message = record["message"]
                timestamp = str(record["timestamp"])
                
                # Quick database info (with fallback for free tier)
                try:
                    db_result = session.run("CALL db.info() YIELD name")
                    db_info = db_result.single()
                except:
                    db_info = {"name": "neo4j"}
                
            driver.close()
            
            return ConnectionResult(
                success=True,
                message=f"{message} at {timestamp}",
                connection_info={
                    "database_name": db_info["name"] if db_info else "neo4j",
                    "timestamp": timestamp
                }
            )
            
        except AuthError as e:
            return ConnectionResult(
                success=False,
                message="Authentication failed - check username and password",
                error_type="AuthError"
            )
        except ServiceUnavailable as e:
            return ConnectionResult(
                success=False,
                message="Service unavailable - check URI and network connectivity",
                error_type="ServiceUnavailable",
                retry_after=30
            )
        except Exception as e:
            return ConnectionResult(
                success=False,
                message=f"Connection test failed: {str(e)}",
                error_type=type(e).__name__
            )
    
    def get_driver(self) -> Driver:
        """Get the current driver instance, connecting if necessary."""
        if self._driver is None or self._status != ConnectionStatus.CONNECTED:
            self.connect()
        return self._driver
    
    def get_status(self) -> Dict[str, Any]:
        """Get current connection status and information."""
        return {
            "status": self._status.value,
            "connected": self._status == ConnectionStatus.CONNECTED,
            "last_error": self._last_error,
            "retry_count": self._retry_count,
            "connection_info": self._connection_info
        }
    
    def health_check(self) -> ConnectionResult:
        """Perform health check on existing connection."""
        if not self._driver or self._status != ConnectionStatus.CONNECTED:
            return ConnectionResult(False, "Not connected")
        
        try:
            with self._driver.session() as session:
                result = session.run("RETURN 1 as health_check, datetime() as timestamp")
                record = result.single()
                timestamp = str(record["timestamp"])
                
            return ConnectionResult(
                success=True,
                message=f"Health check passed at {timestamp}",
                connection_info={"timestamp": timestamp}
            )
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            self._last_error = str(e)
            return ConnectionResult(False, f"Health check failed: {str(e)}")
    
    def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            try:
                self._driver.close()
                logger.info("Neo4j connection closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Neo4j connection: {str(e)}")
            finally:
                self._driver = None
                self._status = ConnectionStatus.DISCONNECTED
                self._connection_info = {}
                self._last_error = None

class QdrantConnection:
    """Manages Qdrant cloud database connection with enhanced validation and error handling."""
    
    def __init__(self):
        self.config = get_config()
        self._client: Optional[QdrantClient] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._last_error: Optional[str] = None
        self._connection_info: Dict[str, Any] = {}
        self._retry_count = 0
        self._max_retries = 3
        self._retry_delay = 2  # seconds
    
    def connect(self, max_retries: int = 3) -> QdrantClient:
        """
        Establish connection to Qdrant cloud instance with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            
        Returns:
            Qdrant client instance
            
        Raises:
            ConnectionError: If connection fails after all retries
        """
        self._status = ConnectionStatus.CONNECTING
        self._retry_count = 0
        
        for attempt in range(max_retries):
            try:
                self._retry_count = attempt + 1
                logger.info(f"Qdrant connection attempt {self._retry_count}/{max_retries}")
                
                self._client = QdrantClient(
                    url=self.config.database.qdrant_url,
                    api_key=self.config.database.qdrant_api_key,
                    timeout=30
                )
                
                # Test connection with comprehensive validation
                validation_result = self._validate_connection()
                if not validation_result.success:
                    self._client.close()
                    self._client = None
                    raise ConnectionError(validation_result.message)
                
                self._status = ConnectionStatus.CONNECTED
                self._last_error = None
                self._connection_info = validation_result.connection_info or {}
                
                logger.info(f"Successfully connected to Qdrant cloud instance on attempt {self._retry_count}")
                return self._client
                
            except UnexpectedResponse as e:
                self._last_error = str(e)
                error_type = type(e).__name__
                logger.warning(f"Qdrant connection attempt {self._retry_count} failed ({error_type}): {str(e)}")
                
                if attempt < max_retries - 1:
                    sleep_time = self._retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self._status = ConnectionStatus.ERROR
                    raise ConnectionError(f"Qdrant connection failed after {max_retries} attempts: {str(e)}")
                    
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Unexpected error during Qdrant connection: {str(e)}")
                self._status = ConnectionStatus.ERROR
                raise ConnectionError(f"Qdrant connection failed: {str(e)}")
    
    def _validate_connection(self) -> ConnectionResult:
        """
        Validate Qdrant connection with comprehensive checks.
        
        Returns:
            ConnectionResult with validation details
        """
        try:
            # Test basic connectivity by getting collections
            collections_response = self._client.get_collections()
            collections = collections_response.collections
            
            # Get cluster information if available
            try:
                cluster_info = self._client.cluster_info()
                cluster_status = cluster_info.status if hasattr(cluster_info, 'status') else 'unknown'
            except:
                cluster_status = 'single_node'
            
            # Test basic operations
            test_collection_name = "connection_test"
            try:
                # Try to get collection info (will fail if doesn't exist, which is fine)
                self._client.get_collection(test_collection_name)
                test_collection_exists = True
            except:
                test_collection_exists = False
            
            connection_info = {
                "collections_count": len(collections),
                "collections": [col.name for col in collections],
                "cluster_status": cluster_status,
                "url": self.config.database.qdrant_url,
                "test_collection_exists": test_collection_exists
            }
            
            return ConnectionResult(
                success=True,
                message="Qdrant connection validated successfully",
                connection_info=connection_info
            )
            
        except Exception as e:
            return ConnectionResult(False, f"Connection validation failed: {str(e)}")
    
    def test_connection(self) -> ConnectionResult:
        """
        Test Qdrant connection without storing the client.
        
        Returns:
            ConnectionResult with test details
        """
        try:
            client = QdrantClient(
                url=self.config.database.qdrant_url,
                api_key=self.config.database.qdrant_api_key,
                timeout=10
            )
            
            # Basic connectivity test
            collections_response = client.get_collections()
            collections = collections_response.collections
            
            # Get cluster info if available
            try:
                cluster_info = client.cluster_info()
                cluster_status = cluster_info.status if hasattr(cluster_info, 'status') else 'available'
            except:
                cluster_status = 'single_node'
            
            client.close()
            
            return ConnectionResult(
                success=True,
                message=f"Qdrant connection successful. Collections: {len(collections)}",
                connection_info={
                    "collections_count": len(collections),
                    "collections": [col.name for col in collections],
                    "cluster_status": cluster_status,
                    "url": self.config.database.qdrant_url
                }
            )
            
        except UnexpectedResponse as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                return ConnectionResult(
                    success=False,
                    message="Authentication failed - check API key",
                    error_type="AuthenticationError"
                )
            elif "404" in str(e) or "Not Found" in str(e):
                return ConnectionResult(
                    success=False,
                    message="Service not found - check URL",
                    error_type="NotFoundError"
                )
            else:
                return ConnectionResult(
                    success=False,
                    message=f"Qdrant API error: {str(e)}",
                    error_type="UnexpectedResponse"
                )
        except Exception as e:
            if "timeout" in str(e).lower():
                return ConnectionResult(
                    success=False,
                    message="Connection timeout - check network connectivity",
                    error_type="TimeoutError",
                    retry_after=30
                )
            else:
                return ConnectionResult(
                    success=False,
                    message=f"Connection test failed: {str(e)}",
                    error_type=type(e).__name__
                )
    
    def get_client(self) -> QdrantClient:
        """Get the current client instance, connecting if necessary."""
        if self._client is None or self._status != ConnectionStatus.CONNECTED:
            self.connect()
        return self._client
    
    def get_status(self) -> Dict[str, Any]:
        """Get current connection status and information."""
        return {
            "status": self._status.value,
            "connected": self._status == ConnectionStatus.CONNECTED,
            "last_error": self._last_error,
            "retry_count": self._retry_count,
            "connection_info": self._connection_info
        }
    
    def health_check(self) -> ConnectionResult:
        """Perform health check on existing connection."""
        if not self._client or self._status != ConnectionStatus.CONNECTED:
            return ConnectionResult(False, "Not connected")
        
        try:
            collections_response = self._client.get_collections()
            collections_count = len(collections_response.collections)
            
            return ConnectionResult(
                success=True,
                message=f"Health check passed. Collections: {collections_count}",
                connection_info={"collections_count": collections_count}
            )
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            self._last_error = str(e)
            return ConnectionResult(False, f"Health check failed: {str(e)}")
    
    def close(self):
        """Close the Qdrant connection."""
        if self._client:
            try:
                self._client.close()
                logger.info("Qdrant connection closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Qdrant connection: {str(e)}")
            finally:
                self._client = None
                self._status = ConnectionStatus.DISCONNECTED
                self._connection_info = {}
                self._last_error = None

class DatabaseManager:
    """Manages all database connections and provides unified interface with comprehensive startup validation."""
    
    def __init__(self):
        self.neo4j = Neo4jConnection()
        self.qdrant = QdrantConnection()
        self._initialized = False
        self._startup_time: Optional[str] = None
        self._initialization_errors: List[str] = []
    
    async def initialize(self, validate_only: bool = False) -> Dict[str, Any]:
        """
        Initialize all database connections with comprehensive validation.
        
        Args:
            validate_only: If True, only test connections without establishing persistent connections
            
        Returns:
            Dictionary with detailed connection status for each database
        """
        logger.info("Starting database initialization...")
        
        status = {
            "neo4j": {
                "connected": False,
                "message": "",
                "error_type": None,
                "connection_info": {},
                "retry_after": None
            },
            "qdrant": {
                "connected": False,
                "message": "",
                "error_type": None,
                "connection_info": {},
                "retry_after": None
            },
            "overall_success": False,
            "initialization_time": None,
            "errors": [],
            "warnings": []
        }
        
        self._initialization_errors = []
        
        # Test Neo4j connection
        logger.info("Testing Neo4j connection...")
        neo4j_result = self.neo4j.test_connection()
        status["neo4j"]["connected"] = neo4j_result.success
        status["neo4j"]["message"] = neo4j_result.message
        status["neo4j"]["error_type"] = neo4j_result.error_type
        status["neo4j"]["connection_info"] = neo4j_result.connection_info or {}
        status["neo4j"]["retry_after"] = neo4j_result.retry_after
        
        if not neo4j_result.success:
            self._initialization_errors.append(f"Neo4j: {neo4j_result.message}")
        
        # Test Qdrant connection
        logger.info("Testing Qdrant connection...")
        qdrant_result = self.qdrant.test_connection()
        status["qdrant"]["connected"] = qdrant_result.success
        status["qdrant"]["message"] = qdrant_result.message
        status["qdrant"]["error_type"] = qdrant_result.error_type
        status["qdrant"]["connection_info"] = qdrant_result.connection_info or {}
        status["qdrant"]["retry_after"] = qdrant_result.retry_after
        
        if not qdrant_result.success:
            self._initialization_errors.append(f"Qdrant: {qdrant_result.message}")
        
        # Overall success if both connections work
        status["overall_success"] = neo4j_result.success and qdrant_result.success
        status["errors"] = self._initialization_errors.copy()
        
        if status["overall_success"] and not validate_only:
            # Actually connect if tests passed and we want persistent connections
            try:
                logger.info("Establishing persistent database connections...")
                self.neo4j.connect()
                self.qdrant.connect()
                self._initialized = True
                self._startup_time = time.strftime("%Y-%m-%d %H:%M:%S")
                status["initialization_time"] = self._startup_time
                logger.info("All database connections initialized successfully")
                
            except Exception as e:
                error_msg = f"Failed to establish persistent connections: {str(e)}"
                logger.error(error_msg)
                status["overall_success"] = False
                status["errors"].append(error_msg)
                self._initialization_errors.append(error_msg)
        
        elif status["overall_success"] and validate_only:
            logger.info("Connection validation completed successfully")
            status["initialization_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add warnings for partial failures
        if neo4j_result.success and not qdrant_result.success:
            status["warnings"].append("Neo4j connected but Qdrant failed - vector search will not be available")
        elif not neo4j_result.success and qdrant_result.success:
            status["warnings"].append("Qdrant connected but Neo4j failed - graph operations will not be available")
        
        return status
    
    def validate_connections(self) -> Dict[str, Any]:
        """
        Validate current database connections with detailed health checks.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        validation = {
            "neo4j": {"valid": False, "message": "", "details": {}},
            "qdrant": {"valid": False, "message": "", "details": {}},
            "overall_valid": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "startup_time": self._startup_time,
            "initialized": self._initialized
        }
        
        # Validate Neo4j with health check
        neo4j_health = self.neo4j.health_check()
        validation["neo4j"]["valid"] = neo4j_health.success
        validation["neo4j"]["message"] = neo4j_health.message
        validation["neo4j"]["details"] = {
            "status": self.neo4j.get_status(),
            "connection_info": neo4j_health.connection_info or {}
        }
        
        # Validate Qdrant with health check
        qdrant_health = self.qdrant.health_check()
        validation["qdrant"]["valid"] = qdrant_health.success
        validation["qdrant"]["message"] = qdrant_health.message
        validation["qdrant"]["details"] = {
            "status": self.qdrant.get_status(),
            "connection_info": qdrant_health.connection_info or {}
        }
        
        validation["overall_valid"] = neo4j_health.success and qdrant_health.success
        
        return validation
    
    def get_startup_validation(self) -> Dict[str, Any]:
        """
        Get comprehensive startup validation for all cloud services.
        
        Returns:
            Dictionary with startup validation results
        """
        try:
            from .config import get_config
        except ImportError:
            from config import get_config
        config = get_config()
        
        validation_result = config.validate_configuration()
        
        # Add database connection status
        if self._initialized:
            db_validation = self.validate_connections()
            validation_result["database_connections"] = {
                "neo4j": db_validation["neo4j"],
                "qdrant": db_validation["qdrant"],
                "overall_valid": db_validation["overall_valid"]
            }
        else:
            validation_result["database_connections"] = {
                "neo4j": {"valid": False, "message": "Not initialized"},
                "qdrant": {"valid": False, "message": "Not initialized"},
                "overall_valid": False
            }
        
        validation_result["initialization_errors"] = self._initialization_errors
        validation_result["startup_time"] = self._startup_time
        validation_result["initialized"] = self._initialized
        
        return validation_result
    
    def is_initialized(self) -> bool:
        """Check if database manager is initialized."""
        return self._initialized
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get a summary of all connection statuses."""
        return {
            "neo4j": self.neo4j.get_status(),
            "qdrant": self.qdrant.get_status(),
            "initialized": self._initialized,
            "startup_time": self._startup_time,
            "errors": self._initialization_errors
        }
    
    def close_all(self):
        """Close all database connections."""
        logger.info("Closing all database connections...")
        self.neo4j.close()
        self.qdrant.close()
        self._initialized = False
        self._startup_time = None
        self._initialization_errors = []
        logger.info("All database connections closed")
    
    @asynccontextmanager
    async def get_neo4j_session(self):
        """Context manager for Neo4j sessions with connection validation."""
        if not self._initialized or self.neo4j._status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Neo4j connection not available")
        
        driver = self.neo4j.get_driver()
        session = driver.session()
        try:
            yield session
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_qdrant_client(self):
        """Context manager for Qdrant client with connection validation."""
        if not self._initialized or self.qdrant._status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Qdrant connection not available")
        
        client = self.qdrant.get_client()
        try:
            yield client
        finally:
            # Qdrant client doesn't need explicit session closing
            pass

# Global database manager instance
db_manager = DatabaseManager()

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager