import os
import sqlite3
import json
try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from werkzeug.security import generate_password_hash
except ImportError:
    def generate_password_hash(pwd):
        import hashlib
        return f"pbkdf2:sha256:600000${hashlib.sha256(pwd.encode()).hexdigest()}"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "backend", "proctorai_local.db")


class SQLiteCursorWrapper:
    def __init__(self, sqlite_cursor):
        self.cursor = sqlite_cursor

    def execute(self, sql, params=None):
        query = sql
        # 1. Handle PostgreSQL specific JSONB / state_change updates
        if "UPDATE action_timeline" in query and "jsonb_set" in query:
            query = "UPDATE action_timeline SET resolved = 1, state_change = '{\"alert\": [\"CREATED\", \"RESOLVED\"]}' WHERE event_uuid = ?"
            if params is not None and len(params) == 1:
                return self.cursor.execute(query, params)

        # 2. Replace %s placeholders with SQLite ? placeholders
        if params is not None:
            query = query.replace("%s", "?")
            # If params is a list or tuple, format JSON objects if any
            if isinstance(params, (list, tuple)):
                converted = []
                for p in params:
                    if isinstance(p, (dict, list)):
                        converted.append(json.dumps(p))
                    else:
                        converted.append(p)
                return self.cursor.execute(query, tuple(converted))
            return self.cursor.execute(query, params)
        else:
            return self.cursor.execute(query)

    def executemany(self, sql, seq_of_params):
        query = sql.replace("%s", "?")
        return self.cursor.executemany(query, seq_of_params)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchmany(self, size=None):
        return self.cursor.fetchmany(size) if size is not None else self.cursor.fetchmany()

    def close(self):
        try:
            self.cursor.close()
        except Exception:
            pass

    @property
    def rowcount(self):
        return self.cursor.rowcount

    @property
    def lastrowid(self):
        return self.cursor.lastrowid

    @property
    def description(self):
        return self.cursor.description

    def __iter__(self):
        return iter(self.cursor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SQLiteConnectionWrapper:
    def __init__(self, sqlite_conn):
        self.conn = sqlite_conn

    def cursor(self):
        return SQLiteCursorWrapper(self.conn.cursor())

    def commit(self):
        return self.conn.commit()

    def rollback(self):
        return self.conn.rollback()

    def close(self):
        try:
            return self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()


def _seed_sqlite(conn):
    """Ensures necessary SQLite tables and default credentials exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS institutions (
                institution_id TEXT PRIMARY KEY,
                institution_name TEXT NOT NULL,
                institution_type TEXT DEFAULT 'University',
                country TEXT DEFAULT 'United States',
                state TEXT DEFAULT '',
                city TEXT DEFAULT '',
                email TEXT DEFAULT '',
                contact TEXT DEFAULT '',
                institution_code TEXT UNIQUE,
                status TEXT DEFAULT 'ACTIVE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                institution_id TEXT,
                student_id TEXT,
                status TEXT DEFAULT 'ACTIVE',
                mfa_secret TEXT DEFAULT 'JBSWY3DPEHPK3PXP',
                mfa_enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE,
                name TEXT,
                face_encoding TEXT,
                arcface_templates TEXT,
                institution_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exam_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                institution_id TEXT,
                risk_score INTEGER,
                direction TEXT,
                status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                username TEXT,
                role TEXT,
                institution_id TEXT,
                action TEXT NOT NULL,
                ip_address TEXT,
                result TEXT NOT NULL,
                details TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uuid TEXT UNIQUE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_str TEXT,
                student_id TEXT,
                student_name TEXT,
                institution_id TEXT,
                category TEXT,
                event_type TEXT,
                title TEXT,
                description TEXT,
                severity TEXT,
                state_change TEXT,
                metadata TEXT,
                resolved INTEGER DEFAULT 0
            );
        """)

        # Ensure default institution exists
        cursor.execute("INSERT OR IGNORE INTO institutions (institution_id, institution_name, institution_code) VALUES ('INST-001', 'Apex Institute of Technology', 'AIT-001');")

        # Seed Faculty Admin (admin@gdsccrce.com / Faculty@123)
        cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = 'admin@gdsccrce.com';")
        if not cursor.fetchone():
            f_hash = generate_password_hash('Faculty@123')
            cursor.execute("""
                INSERT INTO users (name, username, password_hash, role, institution_id, status, mfa_secret, mfa_enabled)
                VALUES ('Faculty Admin', 'admin@gdsccrce.com', ?, 'FACULTY', 'INST-001', 'ACTIVE', 'JBSWY3DPEHPK3PXP', 1);
            """, (f_hash,))

        # Seed Platform Admin (admin / admin)
        cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = 'admin';")
        if not cursor.fetchone():
            a_hash = generate_password_hash('admin')
            cursor.execute("""
                INSERT INTO users (name, username, password_hash, role, institution_id, status, mfa_secret, mfa_enabled)
                VALUES ('Platform Admin', 'admin', ?, 'ADMIN', 'INST-001', 'ACTIVE', 'JBSWY3DPEHPK3PXP', 1);
            """, (a_hash,))

        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"[DB] Error seeding SQLite database: {e}")


import socket
import urllib.parse

_postgres_available = None

def is_postgres_reachable(url):
    global _postgres_available
    if _postgres_available is not None:
        return _postgres_available
    if not url or psycopg2 is None:
        _postgres_available = False
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5432
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        res = s.connect_ex((host, port))
        s.close()
        _postgres_available = (res == 0)
        return _postgres_available
    except Exception:
        _postgres_available = False
        return False


def get_connection(db_url=None):
    """Attempts PostgreSQL connection first; gracefully falls back to wrapped SQLite."""
    url = db_url or os.environ.get("DATABASE_URL", "")
    if is_postgres_reachable(url):
        try:
            return psycopg2.connect(url, connect_timeout=2)
        except Exception:
            pass

    # SQLite fallback
    try:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        _seed_sqlite(conn)
        return SQLiteConnectionWrapper(conn)
    except Exception as e:
        print(f"[DB] SQLite connection fallback error: {e}")
        return None


def init_db(db_url=None):
    """Initializes PostgreSQL schema if available, otherwise initializes & seeds SQLite."""
    url = db_url or os.environ.get("DATABASE_URL", "")
    if is_postgres_reachable(url):
        try:
            conn = psycopg2.connect(url, connect_timeout=2)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS institutions (
                    institution_id TEXT PRIMARY KEY,
                    institution_name VARCHAR(150) NOT NULL,
                    institution_type VARCHAR(50) DEFAULT 'University',
                    country VARCHAR(100) DEFAULT 'United States',
                    state VARCHAR(100) DEFAULT '',
                    city VARCHAR(100) DEFAULT '',
                    email VARCHAR(150) DEFAULT '',
                    contact VARCHAR(50) DEFAULT '',
                    institution_code VARCHAR(50) UNIQUE NOT NULL,
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    institution_id TEXT REFERENCES institutions(institution_id) ON DELETE SET NULL,
                    student_id TEXT,
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    mfa_secret VARCHAR(32) DEFAULT 'JBSWY3DPEHPK3PXP',
                    mfa_enabled INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(50) UNIQUE,
                    name VARCHAR(100),
                    face_encoding TEXT,
                    arcface_templates TEXT,
                    institution_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER,
                    username VARCHAR(50),
                    role VARCHAR(20),
                    institution_id TEXT,
                    action VARCHAR(50) NOT NULL,
                    ip_address TEXT,
                    result VARCHAR(20) NOT NULL,
                    details TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_timeline (
                    id SERIAL PRIMARY KEY,
                    event_uuid VARCHAR(64) UNIQUE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    time_str VARCHAR(20),
                    student_id VARCHAR(50),
                    student_name VARCHAR(100),
                    institution_id TEXT,
                    category VARCHAR(50),
                    event_type VARCHAR(50),
                    title VARCHAR(150),
                    description TEXT,
                    severity VARCHAR(20),
                    state_change TEXT,
                    metadata TEXT,
                    resolved INTEGER DEFAULT 0
                );
            """)
            conn.commit()
            cursor.close()
            conn.close()
            print("[DB] PostgreSQL schema initialized successfully.")
            return True
        except Exception as e:
            print(f"[DB] PostgreSQL initialization skipped, falling back to SQLite: {e}")

    # Initialize SQLite fallback
    try:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        _seed_sqlite(conn)
        conn.close()
        print("[DB] SQLite database initialized and seeded successfully.")
        return True
    except Exception as e:
        print(f"[DB] SQLite initialization error: {e}")
        return False
