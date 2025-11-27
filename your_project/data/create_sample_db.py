"""
Create a small sample SQLite database named northwind.sqlite under data/ for local testing.
This script creates minimal tables and inserts sample rows so the agent and tools can run.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "northwind.sqlite"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Products (
    ProductID INTEGER PRIMARY KEY,
    ProductName TEXT NOT NULL,
    Category TEXT,
    UnitPrice REAL
);

CREATE TABLE IF NOT EXISTS Customers (
    CustomerID INTEGER PRIMARY KEY,
    CompanyName TEXT NOT NULL,
    ContactName TEXT
);

CREATE TABLE IF NOT EXISTS Orders (
    OrderID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    OrderDate TEXT,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);

CREATE TABLE IF NOT EXISTS OrderDetails (
    OrderDetailID INTEGER PRIMARY KEY,
    OrderID INTEGER,
    ProductID INTEGER,
    Quantity INTEGER,
    UnitPrice REAL,
    FOREIGN KEY(OrderID) REFERENCES Orders(OrderID),
    FOREIGN KEY(ProductID) REFERENCES Products(ProductID)
);
"""

SAMPLE_SQL = [
    "INSERT INTO Products (ProductName, Category, UnitPrice) VALUES ('Chai', 'Beverages', 18.0);",
    "INSERT INTO Products (ProductName, Category, UnitPrice) VALUES ('Coffee', 'Beverages', 18.5);",
    "INSERT INTO Products (ProductName, Category, UnitPrice) VALUES ('Aniseed Syrup', 'Condiments', 10.0);",

    "INSERT INTO Customers (CompanyName, ContactName) VALUES ('Acme Corp', 'Alice');",
    "INSERT INTO Customers (CompanyName, ContactName) VALUES ('Big Shop', 'Bob');",

    "INSERT INTO Orders (CustomerID, OrderDate) VALUES (1, '2024-06-01');",
    "INSERT INTO Orders (CustomerID, OrderDate) VALUES (2, '2024-07-15');",

    "INSERT INTO OrderDetails (OrderID, ProductID, Quantity, UnitPrice) VALUES (1, 1, 10, 18.0);",
    "INSERT INTO OrderDetails (OrderID, ProductID, Quantity, UnitPrice) VALUES (1, 3, 2, 10.0);",
    "INSERT INTO OrderDetails (OrderID, ProductID, Quantity, UnitPrice) VALUES (2, 2, 5, 18.5);",
]


def create_db(path: Path = DB_PATH):
    if path.exists():
        print(f"DB already exists at {path} — will replace it for a fresh sample.")
        path.unlink()

    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    for sql in SAMPLE_SQL:
        cur.execute(sql)
    conn.commit()

    print(f"✅ Sample database created at: {path}")
    conn.close()


if __name__ == "__main__":
    create_db()
