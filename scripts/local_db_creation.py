import glob
import os
import sqlite3
import pandas as pd


def create_db(csv_data_dir: str, db_root_dir: str, db_filename: str) -> None:
    csv_files = glob.glob(os.path.join(csv_data_dir, "*.csv"))
    if not os.path.exists(db_root_dir):
        os.makedirs(db_root_dir)
    db_path = os.path.join(db_root_dir, db_filename)
    conn = sqlite3.connect(db_path)
    for file in csv_files:
        print(f"Creating table from {file}")
        df = pd.read_csv(file)
        table_name = os.path.basename(file).replace(".csv", "")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Created table {table_name}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    csv_data_dir = '/Users/arodriguez/Downloads/coherent-11-07-2022/csv'
    db_dir = '/Users/arodriguez/Desktop/FA24-high-risk-project-ai-healthcare/db_dir'
    create_db(csv_data_dir, db_dir, 'coherent_data.db' )
