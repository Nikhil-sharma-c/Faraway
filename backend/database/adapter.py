import os
import psycopg2
import sqlite3

def get_connection(db_url=None):
    """Returns a PostgreSQL connection if db_url is configured, otherwise None."""
    url = db_url or os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    try:
        return psycopg2.connect(url)
    except Exception as e:
        print(f"[DB] PostgreSQL connection unavailable: {e}")
        return None

def init_db(db_url=None):
    """Initializes PostgreSQL schema if available."""
    url = db_url or os.environ.get("DATABASE_URL", "")
    if not url:
        return False
    try:
        conn = psycopg2.connect(url)
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
                mfa_secret VARCHAR(32),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                institution_id TEXT REFERENCES institutions(institution_id) ON DELETE CASCADE,
                embedding BYTEA NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exam_logs (
                log_id SERIAL PRIMARY KEY,
                student_id VARCHAR(50),
                risk_score INTEGER,
                direction VARCHAR(20),
                status VARCHAR(50),
                institution_id TEXT REFERENCES institutions(institution_id) ON DELETE CASCADE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exam_sessions (
                session_id SERIAL PRIMARY KEY,
                institution_id TEXT REFERENCES institutions(institution_id) ON DELETE CASCADE,
                supervisor_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status VARCHAR(20) DEFAULT 'ACTIVE',
                duration_seconds INTEGER DEFAULT 0,
                report_url TEXT
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                audit_id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
                username VARCHAR(50),
                role VARCHAR(20),
                institution_id TEXT,
                action VARCHAR(50) NOT NULL,
                resource_path TEXT,
                status VARCHAR(20) NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeline_events (
                id VARCHAR(32) PRIMARY KEY,
                timestamp VARCHAR(20),
                iso_timestamp VARCHAR(40),
                student_id VARCHAR(50),
                student_name VARCHAR(100),
                institution_id TEXT,
                category VARCHAR(50),
                event_type VARCHAR(50),
                title VARCHAR(150),
                description TEXT,
                severity VARCHAR(20),
                state_change JSONB,
                metadata JSONB,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("[DB] PostgreSQL schema initialized successfully.")
        return True
    except Exception as e:
        print(f"[DB] PostgreSQL initialization skipped: {e}")
        return False
