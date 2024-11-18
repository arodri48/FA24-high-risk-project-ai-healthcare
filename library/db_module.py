import sqlite3

class Db:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        self.conn.commit()

    def __del__(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

    def query(self, sql: str, params=None):
        if params is None:
            self.cursor.execute(sql)
        else:
            self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def execute(self, sql: str, params=None):
        if params is None:
            self.cursor.execute(sql)
        else:
            self.cursor.execute(sql, params)
        self.conn.commit()

    def get_schema(self):
        return self.query("SELECT sql FROM sqlite_master WHERE type='table'")

    def get_table(self, table_name: str):
        return self.query(f"SELECT * FROM {table_name}")

    def get_table_columns(self, table_name: str):
        return [column[1] for column in self.query(f"PRAGMA table_info({table_name})")]

    def get_table_column_types(self, table_name: str):
        return [column[2] for column in self.query(f"PRAGMA table_info({table_name})")]

    def get_table_column_names(self, table_name: str):
        return [column[1] for column in self.query(f"PRAGMA table_info({table_name})")]

    def get_table_column_name_type(self, table_name: str):
        return [(column[1], column[2]) for column in self.query(f"PRAGMA table_info({table_name})")]

    def get_table_column_name_type_dict(self, table_name: str):
        return {column[1]: column[2] for column in self.query(f"PRAGMA table_info({table_name})")}
