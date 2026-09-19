import sqlite3
import pandas as pd

conn = sqlite3.connect('saas_intel.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

try:
    df = pd.read_sql('SELECT * FROM clean_saas_customers LIMIT 5', conn)
    print("Columns:", df.columns.tolist())
    print(df)
except Exception as e:
    print("Error:", e)
