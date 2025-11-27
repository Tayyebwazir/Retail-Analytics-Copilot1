"""
SQLite database tool for executing queries safely.
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class SQLiteExecutionResult:
    """Container for SQL execution results."""
    
    def __init__(self, 
                 success: bool,
                 rows: List[Dict[str, Any]] = None,
                 error: str = None,
                 sql: str = None,
                 columns: List[str] = None):
        self.success = success
        self.rows = rows or []
        self.error = error
        self.sql = sql
        self.columns = columns or []
    
    def __repr__(self):
        if self.success:
            return f"<SQLResult: {len(self.rows)} rows>"
        else:
            return f"<SQLResult: ERROR - {self.error}>"


class SQLiteTool:
    """Safe SQLite query execution with schema introspection."""
    
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Access columns by name
        self._schema_cache = None
        print(f"✅ Connected to database: {self.db_path}")
    
    def get_schema(self, refresh: bool = False) -> str:
        """
        Get database schema in a formatted string.
        
        Returns:
            Human-readable schema description
        """
        if self._schema_cache and not refresh:
            return self._schema_cache
        
        cursor = self.connection.cursor()
        
        # Get all tables (excluding SQLite internal tables)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_lines = ["DATABASE SCHEMA:", "=" * 50]
        
        for table in tables:
            # Get columns for each table
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            schema_lines.append(f"\nTable: {table}")
            schema_lines.append("-" * 40)
            
            for col in columns:
                col_id, name, col_type, not_null, default, pk = col
                pk_marker = " [PRIMARY KEY]" if pk else ""
                null_marker = " NOT NULL" if not_null else ""
                schema_lines.append(f"  - {name}: {col_type}{pk_marker}{null_marker}")
        
        self._schema_cache = "\n".join(schema_lines)
        return self._schema_cache
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def validate_query(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Basic validation to prevent dangerous queries.
        
        Returns:
            (is_valid, error_message)
        """
        sql_upper = sql.upper().strip()
        
        # Block dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Dangerous keyword '{keyword}' not allowed"
        
        # Must be a SELECT query
        if not sql_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, None
    
    def execute_query(self, sql: str) -> SQLiteExecutionResult:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query string (SELECT only)
        
        Returns:
            SQLiteExecutionResult with success status and data
        """
        # Validate query
        is_valid, error_msg = self.validate_query(sql)
        if not is_valid:
            return SQLiteExecutionResult(
                success=False,
                error=error_msg,
                sql=sql
            )
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows_as_dicts = [dict(row) for row in rows]
            
            return SQLiteExecutionResult(
                success=True,
                rows=rows_as_dicts,
                sql=sql,
                columns=columns
            )
        
        except sqlite3.Error as e:
            return SQLiteExecutionResult(
                success=False,
                error=str(e),
                sql=sql
            )
        
        except Exception as e:
            return SQLiteExecutionResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                sql=sql
            )
    
    def test_connection(self) -> bool:
        """Test if database connection is working."""
        try:
            result = self.execute_query("SELECT 1 as test")
            return result.success
        except:
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


# Singleton instance
_db_tool_instance = None


def get_db_tool(db_path: str = "data/northwind.sqlite") -> SQLiteTool:
    """Get or create the global database tool instance."""
    global _db_tool_instance
    if _db_tool_instance is None:
        _db_tool_instance = SQLiteTool(db_path)
    return _db_tool_instance


if __name__ == "__main__":
    # Test the database tool
    db = get_db_tool()
    
    print("\n" + "=" * 60)
    print("DATABASE SCHEMA")
    print("=" * 60)
    print(db.get_schema())
    
    print("\n" + "=" * 60)
    print("TEST QUERY")
    print("=" * 60)
    
    test_sql = """
        SELECT ProductName, UnitPrice 
        FROM Products 
        LIMIT 5
    """
    
    result = db.execute_query(test_sql)
    
    if result.success:
        print(f"✅ Query successful! {len(result.rows)} rows returned")
        for row in result.rows:
            print(f"  {row}")
    else:
        print(f"❌ Query failed: {result.error}")