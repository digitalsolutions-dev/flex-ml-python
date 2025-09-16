import os, mysql.connector
from typing import Any, List, Tuple


def call_proc(name: str, params: Tuple[Any, ...] = ()) -> List[dict]:
    conn = mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", "example"),
        database=os.environ.get("DB_NAME", "flexml"),
    )
    try:
        cur = conn.cursor(dictionary=True)
        cur.callproc(name, params)
        rows = []
        for result in cur.stored_results():
            rows = result.fetchall()
        conn.commit()
        return rows
    finally:
        cur.close();
        conn.close()
