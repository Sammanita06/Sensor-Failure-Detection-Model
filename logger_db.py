import os
import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "model_store/operations_log.db"

def init_db():
    os.makedirs("model_store", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS batch_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, batch_size INTEGER NOT NULL, f1_score REAL NOT NULL, precision_score REAL NOT NULL, recall_score REAL NOT NULL, tier_used TEXT NOT NULL, drift_detected INTEGER NOT NULL)""")
    conn.commit()
    conn.close()

def log_batch_execution(batch_size, f1, precision, recall, tier_used, drift_detected):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO batch_logs (timestamp, batch_size, f1_score, precision_score, recall_score, tier_used, drift_detected) VALUES (?, ?, ?, ?, ?, ?, ?)""", (datetime.now().strftime("%%Y-%%m-%%d %%H:%%M:%%S"), batch_size, float(f1), float(precision), float(recall), tier_used, 1 if drift_detected else 0))
    conn.commit()
    conn.close()

def fetch_historical_logs():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM batch_logs ORDER BY id ASC", conn)
    conn.close()
    return df
