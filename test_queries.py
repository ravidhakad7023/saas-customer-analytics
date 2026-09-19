import sqlite3
import os
import glob
import traceback

conn = sqlite3.connect('saas_intel.db')
sql_files = sorted(glob.glob('queries/*.sql'))

for f in sql_files:
    print(f"\n--- Running {f} ---")
    with open(f, 'r') as file:
        query = file.read()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(3)
        for r in rows:
            print(r)
    except Exception as e:
        print(f"FAILED: {e}")
        # traceback.print_exc()

conn.close()
