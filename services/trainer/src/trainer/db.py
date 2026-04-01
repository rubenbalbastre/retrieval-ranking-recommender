from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from trainer.config import settings


def get_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        row_factory=dict_row,
    )


def execute_sql_file(path: str) -> None:
    with get_connection() as conn, open(path, "r", encoding="utf-8") as f:
        conn.execute(f.read())
        conn.commit()
