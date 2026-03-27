"""
SQLite experiment log per spec v0.4.
Atomic writes per trial for crash recovery.
"""
import json
import sqlite3
import os

import config


def init_db(db_path: str = None) -> sqlite3.Connection:
    """Initialize the experiment database with the spec's schema."""
    if db_path is None:
        db_path = config.DB_PATH

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trials (
            trial_id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            intervention_spec TEXT NOT NULL,
            is_degenerate BOOLEAN NOT NULL DEFAULT 0,
            accuracy REAL,
            accuracy_delta REAL,
            items_flipped_to_correct INTEGER,
            items_flipped_to_incorrect INTEGER,
            mcnemar_p_value REAL,
            bh_significant BOOLEAN,
            tier INTEGER,
            tier_justification TEXT,
            wall_clock_seconds REAL NOT NULL,
            vram_peak_bytes INTEGER,
            timestamp_utc TEXT NOT NULL,
            random_seed INTEGER
        );

        CREATE TABLE IF NOT EXISTS item_results (
            trial_id TEXT NOT NULL REFERENCES trials(trial_id),
            item_id TEXT NOT NULL,
            baseline_correct BOOLEAN NOT NULL,
            intervention_correct BOOLEAN,
            baseline_logprobs TEXT,
            intervention_logprobs TEXT,
            PRIMARY KEY (trial_id, item_id)
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def save_trial(conn: sqlite3.Connection, trial: dict, item_results: list[dict]):
    """Atomically save a trial and its item results."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO trials
            (trial_id, category, intervention_spec, is_degenerate, accuracy,
             accuracy_delta, items_flipped_to_correct, items_flipped_to_incorrect,
             mcnemar_p_value, bh_significant, tier, tier_justification,
             wall_clock_seconds, vram_peak_bytes, timestamp_utc, random_seed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial['trial_id'],
            trial['category'],
            json.dumps(trial['intervention_spec']),
            trial.get('is_degenerate', False),
            trial.get('accuracy'),
            trial.get('accuracy_delta'),
            trial.get('items_flipped_to_correct'),
            trial.get('items_flipped_to_incorrect'),
            trial.get('mcnemar_p_value'),
            trial.get('bh_significant'),
            trial.get('tier'),
            trial.get('tier_justification'),
            trial['wall_clock_seconds'],
            trial.get('vram_peak_bytes'),
            trial['timestamp_utc'],
            trial.get('random_seed'),
        ))

        # Batch insert item results
        if item_results:
            cursor.executemany("""
                INSERT OR REPLACE INTO item_results
                (trial_id, item_id, baseline_correct, intervention_correct,
                 baseline_logprobs, intervention_logprobs)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (
                    r['trial_id'],
                    r['item_id'],
                    r['baseline_correct'],
                    r.get('intervention_correct'),
                    json.dumps(r.get('baseline_logprobs')) if r.get('baseline_logprobs') else None,
                    json.dumps(r.get('intervention_logprobs')) if r.get('intervention_logprobs') else None,
                )
                for r in item_results
            ])

        conn.commit()
    except Exception:
        conn.rollback()
        raise


def save_metadata(conn: sqlite3.Connection, key: str, value):
    """Save a metadata key-value pair."""
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (key, json.dumps(value, default=str))
    )
    conn.commit()


def get_completed_trial_ids(conn: sqlite3.Connection) -> set[str]:
    """Get all completed trial IDs for crash recovery."""
    cursor = conn.execute("SELECT trial_id FROM trials")
    return {row[0] for row in cursor.fetchall()}


def get_trials_by_category(conn: sqlite3.Connection, category: str = None) -> list[dict]:
    """Get trials, optionally filtered by category."""
    if category:
        cursor = conn.execute(
            "SELECT * FROM trials WHERE category = ? ORDER BY trial_id",
            (category,)
        )
    else:
        cursor = conn.execute("SELECT * FROM trials ORDER BY trial_id")

    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_non_degenerate_trials(conn: sqlite3.Connection) -> list[dict]:
    """Get all non-degenerate trials (for BH correction pool)."""
    cursor = conn.execute(
        "SELECT * FROM trials WHERE is_degenerate = 0 ORDER BY trial_id"
    )
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
