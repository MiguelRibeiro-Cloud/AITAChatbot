import os

import psycopg


class CounterConfigError(RuntimeError):
    """Raised when the counter database configuration is incomplete."""


class CounterDatabaseError(RuntimeError):
    """Raised when the counter database call fails."""


REQUIRED_ENV_VARS = (
    "POSTGRES_HOST",
    "POSTGRES_DATABASE",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


def _connection_config():
    missing = [name for name in REQUIRED_ENV_VARS if not (os.environ.get(name) or "").strip()]
    if missing:
        raise CounterConfigError(f"Missing required counter database setting(s): {', '.join(missing)}")

    port_value = (os.environ.get("POSTGRES_PORT") or "5432").strip()
    try:
        port = int(port_value)
    except ValueError as exc:
        raise CounterConfigError("POSTGRES_PORT must be an integer") from exc

    return {
        "host": os.environ["POSTGRES_HOST"],
        "port": port,
        "dbname": os.environ["POSTGRES_DATABASE"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "sslmode": os.environ.get("POSTGRES_SSLMODE", "require"),
        "connect_timeout": 5,
        "options": "-c statement_timeout=5000",
    }


def _fetch_counter(sql):
    try:
        with psycopg.connect(**_connection_config()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
            if not row or row[0] is None:
                raise CounterDatabaseError("Counter function returned no value")
            return int(row[0])
    except CounterConfigError:
        raise
    except CounterDatabaseError:
        raise
    except Exception as exc:
        raise CounterDatabaseError("Counter database operation failed") from exc


def get_cases_heard():
    return _fetch_counter("SELECT counter.get_cases_heard();")


def increment_cases_heard():
    return _fetch_counter("SELECT counter.increment_cases_heard();")
