"""Database connection utility supporting MySQL, SQL Server, and Oracle."""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import pyodbc
    SQLSERVER_AVAILABLE = True
except ImportError:
    SQLSERVER_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_type: str
    host: str
    port: int
    instance: Optional[str]
    database: str
    username: str
    password: str


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabaseQueryError(Exception):
    """Raised when query execution fails."""
    pass


def get_database_config() -> DatabaseConfig:
    """Get database configuration from environment variables."""
    db_type = os.getenv('DB_TYPE', 'mysql').lower()
    host = os.getenv('DB_HOST', 'localhost')
    port = int(os.getenv('DB_PORT', '3306'))
    instance = os.getenv('DB_INSTANCE', '')
    database = os.getenv('DB_DATABASE', '')
    username = os.getenv('DB_USERNAME', '')
    password = os.getenv('DB_PASSWORD', '')
    
    if not database or not username:
        raise ValueError("DB_DATABASE and DB_USERNAME must be set in environment variables")
    
    return DatabaseConfig(
        db_type=db_type,
        host=host,
        port=port,
        instance=instance if instance else None,
        database=database,
        username=username,
        password=password
    )


class DatabaseConnector:
    """Database connector supporting MySQL, SQL Server, and Oracle."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database connector with configuration."""
        self.config = config or get_database_config()
        self.connection = None
        
    def _get_mysql_connection(self):
        """Create MySQL connection."""
        if not MYSQL_AVAILABLE:
            raise DatabaseConnectionError("PyMySQL not installed. Install with: pip install PyMySQL")
        
        try:
            return pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                autocommit=True,
                cursorclass=pymysql.cursors.DictCursor
            )
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to MySQL: {str(e)}")
    
    def _get_sqlserver_connection(self):
        """Create SQL Server connection."""
        if not SQLSERVER_AVAILABLE:
            raise DatabaseConnectionError("pyodbc not installed. Install with: pip install pyodbc")
        
        try:
            # Build connection string for SQL Server
            if self.config.instance:
                server = f"{self.config.host}\\{self.config.instance}"
            else:
                server = f"{self.config.host},{self.config.port}"
                
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={self.config.database};"
                f"UID={self.config.username};"
                f"PWD={self.config.password};"
                f"TrustServerCertificate=yes;"
            )
            
            return pyodbc.connect(conn_str)
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to SQL Server: {str(e)}")
    
    def _get_oracle_connection(self):
        """Create Oracle connection."""
        if not ORACLE_AVAILABLE:
            raise DatabaseConnectionError("cx_Oracle not installed. Install with: pip install cx_Oracle")
        
        try:
            # Build Oracle connection string
            if self.config.instance:
                dsn = f"{self.config.host}:{self.config.port}/{self.config.instance}"
            else:
                dsn = f"{self.config.host}:{self.config.port}/{self.config.database}"
                
            return cx_Oracle.connect(
                user=self.config.username,
                password=self.config.password,
                dsn=dsn
            )
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to Oracle: {str(e)}")
    
    def connect(self):
        """Establish database connection based on configured type."""
        if self.connection:
            return self.connection
            
        logger.info(f"Connecting to {self.config.db_type} database at {self.config.host}:{self.config.port}")
        
        if self.config.db_type == 'mysql':
            self.connection = self._get_mysql_connection()
        elif self.config.db_type in ['sqlserver', 'mssql']:
            self.connection = self._get_sqlserver_connection()
        elif self.config.db_type == 'oracle':
            self.connection = self._get_oracle_connection()
        else:
            raise DatabaseConnectionError(f"Unsupported database type: {self.config.db_type}")
        
        logger.info(f"Successfully connected to {self.config.db_type} database")
        return self.connection
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")
            finally:
                self.connection = None
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute SQL query and return results as list of dictionaries and column names."""
        if not self.connection:
            self.connect()
        
        try:
            logger.info(f"Executing query: {query[:100]}...")
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Get column names
            if self.config.db_type == 'mysql':
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
            elif self.config.db_type in ['sqlserver', 'mssql']:
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            elif self.config.db_type == 'oracle':
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                raise DatabaseQueryError(f"Unsupported database type: {self.config.db_type}")
            
            cursor.close()
            
            logger.info(f"Query executed successfully, returned {len(results)} rows")
            return results, columns
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise DatabaseQueryError(f"Failed to execute query: {str(e)}")
    
    def execute_query_to_dataframe(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as pandas DataFrame."""
        results, columns = self.execute_query(query)
        
        if not results:
            return pd.DataFrame(columns=columns)
        
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=columns)
        logger.info(f"Query returned DataFrame with shape {df.shape}")
        return df
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return connection info."""
        try:
            self.connect()
            
            # Execute a simple test query
            test_query = "SELECT 1 as test_value"
            if self.config.db_type == 'oracle':
                test_query = "SELECT 1 as test_value FROM dual"
            
            results, _ = self.execute_query(test_query)
            
            return {
                "success": True,
                "db_type": self.config.db_type,
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "test_result": results[0] if results else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "db_type": self.config.db_type,
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database
            }
        finally:
            self.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def format_query_results_as_table(results: List[Dict[str, Any]], columns: List[str]) -> str:
    """Format query results as a readable table string."""
    if not results:
        return "No results returned."
    
    # Convert to DataFrame for better formatting
    df = pd.DataFrame(results, columns=columns)
    
    # Format the DataFrame as a string table
    table_str = df.to_string(index=False, max_rows=100, max_cols=20)
    
    # Add summary information
    summary = f"\nQuery Results Summary:\n"
    summary += f"- Rows returned: {len(results)}\n"
    summary += f"- Columns: {len(columns)}\n"
    
    if len(results) > 100:
        summary += f"- Note: Only first 100 rows displayed\n"
    
    return table_str + "\n" + summary


def get_available_database_drivers() -> Dict[str, bool]:
    """Check which database drivers are available."""
    return {
        "mysql": MYSQL_AVAILABLE,
        "sqlserver": SQLSERVER_AVAILABLE,
        "oracle": ORACLE_AVAILABLE
    }