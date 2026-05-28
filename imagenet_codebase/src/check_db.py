# check_db.py
import sqlite3
import os

db_path = "data/conceptnet_local.db"

print(f"Checking database at: {db_path}")
print(f"File exists: {os.path.exists(db_path)}")
# if os.path.exists(db_path):
#     print(f"File size: {os.getsize(db_path) / (1024*1024):.2f} MB")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query all table names inside the SQLite master catalog
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found in database: {tables}")
    
    if tables:
        primary_table = tables[0][0]
        cursor.execute(f"PRAGMA table_info({primary_table});")
        columns = cursor.fetchall()
        print(f"Columns in table '{primary_table}': {[col[1] for col in columns]}")
    else:
        print("[Warning] This database file contains zero tables! The file is blank.")
        
    conn.close()
except Exception as e:
    print(f"Error accessing DB: {e}")