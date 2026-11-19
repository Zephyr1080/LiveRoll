"""SQLite helpers and the in-memory list of known faces."""
import sqlite3
import numpy as np
import pandas as pd
from config import DB_PATH


def run_sql(query, params=(), fetch=False):
    """Run one SQL statement and return rows if fetch=True."""
    db = sqlite3.connect(DB_PATH)
    try:
        cur = db.execute(query, params)
        rows = cur.fetchall() if fetch else None
        db.commit()
        return rows
    finally:
        db.close()


def query_df(query, params=()):
    """Run a SELECT and return the result as a pandas DataFrame."""
    db = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, db, params=params)
    finally:
        db.close()


def init_db():
    run_sql("""CREATE TABLE IF NOT EXISTS students(
                 reg_no TEXT PRIMARY KEY, name TEXT NOT NULL, encoding BLOB NOT NULL)""")
    run_sql("""CREATE TABLE IF NOT EXISTS attendance(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 reg_no TEXT, name TEXT, date TEXT, time TEXT, status TEXT,
                 liveness TEXT DEFAULT 'Not checked',
                 UNIQUE(reg_no, date))""")          # UNIQUE blocks duplicate entries
    # Upgrade an older database that was created before the liveness column existed
    cols = [r[1] for r in run_sql("PRAGMA table_info(attendance)", fetch=True)]
    if "liveness" not in cols:
        run_sql("ALTER TABLE attendance ADD COLUMN liveness TEXT DEFAULT 'Not checked'")


# Known faces are kept in memory while the program runs.
# These lists are modified in place, so other modules can import them directly.
known_encodings, known_names, known_regs = [], [], []


def load_known_faces():
    """Load all registered students from the database into the lists."""
    known_encodings.clear(); known_names.clear(); known_regs.clear()
    for reg, name, blob in run_sql("SELECT reg_no, name, encoding FROM students", fetch=True):
        known_regs.append(reg)
        known_names.append(name)
        known_encodings.append(np.frombuffer(blob, dtype=np.float64))


def setup():
    init_db()
    load_known_faces()
    print("Database ready. Registered students:", len(known_regs))
