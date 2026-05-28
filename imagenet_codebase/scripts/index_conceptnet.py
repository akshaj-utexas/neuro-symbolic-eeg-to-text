"""
scripts/index_conceptnet.py
Optimized Version: Uses json.loads and SQLite PRAGMAs for 10x faster indexing.
"""
import sqlite3
import pandas as pd
import json
from tqdm import tqdm
import os

def build_conceptnet_db(csv_path, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Performance Tuning ---
    # Memory-mapped I/O and synchronous off can speed up writes significantly
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")

    cursor.execute("DROP TABLE IF EXISTS assertions")
    cursor.execute("""
        CREATE TABLE assertions (
            start_node TEXT,
            end_node TEXT,
            relation TEXT,
            weight REAL
        )
    """)
    
    print(f"--- Indexing ConceptNet assertions from {csv_path} ---")
    
    chunk_size = 100000 # Smaller chunks can sometimes help with memory pressure
    reader = pd.read_csv(csv_path, sep='\t', header=None, chunksize=chunk_size, 
                         names=['uri', 'rel', 'start', 'end', 'metadata'])

    for chunk in tqdm(reader):
        # 1. Fast English-only filter
        mask = chunk['start'].str.startswith('/c/en/') & chunk['end'].str.startswith('/c/en/')
        valid_data = chunk[mask].copy()
        
        # 2. Replaced eval() with a faster extraction method
        # Most metadata weights are simple. We can use a fast string split or json.loads
        def get_weight(meta_str):
            try:
                # ConceptNet metadata is valid JSON
                return json.loads(meta_str).get('weight', 1.0)
            except:
                return 1.0

        valid_data['weight'] = valid_data['metadata'].apply(get_weight)
        
        to_insert = valid_data[['start', 'end', 'rel', 'weight']].values.tolist()
        cursor.executemany("INSERT INTO assertions VALUES (?, ?, ?, ?)", to_insert)
        conn.commit()

    print("--- Finalizing: Creating Indices (This will take a few minutes) ---")
    # Indices must only be created AFTER data insertion for speed
    cursor.execute("CREATE INDEX idx_start ON assertions(start_node)")
    cursor.execute("CREATE INDEX idx_end ON assertions(end_node)")
    conn.commit()
    conn.close()
    print(f"Local ConceptNet DB ready at {db_path}")
if __name__ == "__main__":
    # Path to the downloaded 'conceptnet-assertions-5.7.0.csv'
    build_conceptnet_db("data/conceptnet-assertions-5.7.0.csv", "data/conceptnet_local.db")