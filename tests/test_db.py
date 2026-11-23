"""Unit tests for spectra.db generic async SQLite base."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from spectra import Database, ExecuteResult, Transaction


def _list_dir(path: Path) -> list[Path]:
    return list(path.iterdir())


async def test_lifecycle_context_manager_and_no_business_tables(tmp_path: Path) -> None:
    """Connect creates parent dirs, initializes only _migrations, and closes cleanly."""
    db_path = tmp_path / "nested" / "dir" / "bot.db"
    db = Database(db_path)
    assert db.db_path == db_path

    async with db as connected:
        assert connected is db
        assert db_path.exists()
        tables = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        assert [row["name"] for row in tables] == ["_migrations"]

    # After context exit, database is closed and rejects operations.
    with pytest.raises(RuntimeError, match="not connected"):
        await db.fetchone("SELECT 1")

    # close() is idempotent.
    await db.close()
    await db.close()


async def test_memory_db_does_not_create_filesystem_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using ':memory:' does not create files or directories in cwd."""
    monkeypatch.chdir(tmp_path)
    async with Database(":memory:") as db:
        row = await db.fetchone("SELECT 42 AS answer")
        assert row is not None
        assert row["answer"] == 42

    assert await asyncio.to_thread(_list_dir, tmp_path) == []


async def test_connect_idempotent_and_concurrent(tmp_path: Path) -> None:
    """Repeated or concurrent connect() calls share a single connection."""
    db = Database(tmp_path / "concurrent_connect.db")
    try:
        await asyncio.gather(db.connect(), db.connect(), db.connect())
        first_conn = db._conn
        assert first_conn is not None
        await db.connect()
        assert db._conn is first_conn
    finally:
        await db.close()


async def test_connect_closes_connection_on_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If initialization fails partway in connect(), the raw connection is closed."""
    real_connect = aiosqlite.connect
    close_calls = 0
    created_conns: list[aiosqlite.Connection] = []

    async def failing_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        conn = await real_connect(*args, **kwargs)
        created_conns.append(conn)
        original_execute = conn.execute
        original_close = conn.close
        call_count = 0

        def flaky_execute(sql: str, *exec_args: object, **exec_kwargs: object):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("boom during pragma")
            return original_execute(sql, *exec_args, **exec_kwargs)

        async def tracked_close() -> None:
            nonlocal close_calls
            close_calls += 1
            await original_close()

        conn.execute = flaky_execute  # type: ignore[method-assign]
        conn.close = tracked_close  # type: ignore[method-assign]
        return conn

    monkeypatch.setattr(aiosqlite, "connect", failing_connect)

    db = Database(tmp_path / "fail_init.db")
    with pytest.raises(RuntimeError, match="boom during pragma"):
        await db.connect()

    assert db._conn is None
    assert len(created_conns) == 1
    assert close_calls == 1
    with pytest.raises(ValueError, match="no active connection"):
        await created_conns[0].commit()


async def test_connect_closes_connection_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation (BaseException) during connect() still closes the connection."""
    real_connect = aiosqlite.connect
    close_calls = 0
    created_conns: list[aiosqlite.Connection] = []

    async def cancelling_connect(
        *args: object,
        **kwargs: object,
    ) -> aiosqlite.Connection:
        conn = await real_connect(*args, **kwargs)
        created_conns.append(conn)
        original_close = conn.close

        async def tracked_close() -> None:
            nonlocal close_calls
            close_calls += 1
            await original_close()

        conn.execute = AsyncMock(side_effect=asyncio.CancelledError())  # type: ignore[method-assign]
        conn.close = tracked_close  # type: ignore[method-assign]
        return conn

    monkeypatch.setattr(aiosqlite, "connect", cancelling_connect)

    db = Database(tmp_path / "cancel_init.db")
    with pytest.raises(asyncio.CancelledError):
        await db.connect()

    assert db._conn is None
    assert len(created_conns) == 1
    assert close_calls == 1
    with pytest.raises(ValueError, match="no active connection"):
        await created_conns[0].commit()


async def test_pragmas_active_on_file_db(tmp_path: Path) -> None:
    """File-backed DB enables WAL, foreign_keys, busy_timeout, NORMAL, MEMORY."""
    async with Database(tmp_path / "pragmas.db") as db:
        journal = await db.fetchone("PRAGMA journal_mode")
        busy_timeout = await db.fetchone("PRAGMA busy_timeout")
        synchronous = await db.fetchone("PRAGMA synchronous")
        foreign_keys = await db.fetchone("PRAGMA foreign_keys")
        temp_store = await db.fetchone("PRAGMA temp_store")

        assert journal is not None and journal[0].lower() == "wal"
        assert busy_timeout is not None and busy_timeout[0] == 30000
        assert synchronous is not None and synchronous[0] == 1  # NORMAL
        assert foreign_keys is not None and foreign_keys[0] == 1  # ON
        assert temp_store is not None and temp_store[0] == 2  # MEMORY

        # Verify foreign key constraints are actually enforced.
        await db.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
        await db.execute(
            "CREATE TABLE children ("
            "id INTEGER PRIMARY KEY, "
            "parent_id INTEGER NOT NULL REFERENCES parents(id)"
            ")"
        )
        with pytest.raises(aiosqlite.IntegrityError):
            await db.execute("INSERT INTO children (parent_id) VALUES (?)", (999,))


async def test_methods_raise_before_connect_and_after_close(tmp_path: Path) -> None:
    """All query/write/tx/migrate methods raise RuntimeError when disconnected."""
    db = Database(tmp_path / "disconnected.db")

    async def assert_all_fail() -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            await db.fetchone("SELECT 1")
        with pytest.raises(RuntimeError, match="not connected"):
            await db.fetchall("SELECT 1")
        with pytest.raises(RuntimeError, match="not connected"):
            await db.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="not connected"):
            async with db.transaction():
                pass
        with pytest.raises(RuntimeError, match="not connected"):
            await db.migrate("sample", ["CREATE TABLE t (id INTEGER PRIMARY KEY)"])

    await assert_all_fail()
    await db.connect()
    await db.close()
    await assert_all_fail()


async def test_execute_fetchone_fetchall_and_execute_result(tmp_path: Path) -> None:
    """execute/fetchone/fetchall return Row objects and frozen ExecuteResult."""
    async with Database(tmp_path / "crud.db") as db:
        await db.execute(
            "CREATE TABLE items ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL"
            ")"
        )

        res1 = await db.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
        assert isinstance(res1, ExecuteResult)
        assert res1.lastrowid == 1
        assert res1.rowcount == 1
        with pytest.raises(FrozenInstanceError):
            res1.rowcount = 99  # type: ignore[misc]

        res2 = await db.execute("INSERT INTO items (name) VALUES (?)", ("beta",))
        assert res2.lastrowid == 2
        assert res2.rowcount == 1

        row = await db.fetchone("SELECT id, name FROM items WHERE id = ?", (1,))
        assert row is not None
        assert row["id"] == 1
        assert row["name"] == "alpha"

        missing = await db.fetchone("SELECT id FROM items WHERE id = ?", (999,))
        assert missing is None

        rows = await db.fetchall("SELECT id, name FROM items ORDER BY id")
        assert isinstance(rows, list)
        assert [(r["id"], r["name"]) for r in rows] == [(1, "alpha"), (2, "beta")]

        upd = await db.execute(
            "UPDATE items SET name = ? WHERE id >= ?",
            ("updated", 1),
        )
        assert upd.rowcount == 2

        deleted = await db.execute("DELETE FROM items WHERE id = ?", (2,))
        assert deleted.rowcount == 1


async def test_transaction_commit_and_rollback_on_exception(tmp_path: Path) -> None:
    """transaction() commits on normal exit and rolls back on exception."""
    db_path = tmp_path / "tx.db"
    async with Database(db_path) as db:
        await db.execute(
            "CREATE TABLE ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, amount INTEGER)"
        )

        async with db.transaction() as tx:
            assert isinstance(tx, Transaction)
            r1 = await tx.execute("INSERT INTO ledger (amount) VALUES (?)", (100,))
            r2 = await tx.execute("INSERT INTO ledger (amount) VALUES (?)", (200,))
            assert r1.lastrowid == 1
            assert r2.lastrowid == 2
            in_tx = await tx.fetchall("SELECT amount FROM ledger ORDER BY id")
            assert [r["amount"] for r in in_tx] == [100, 200]
            first = await tx.fetchone("SELECT amount FROM ledger WHERE id = 1")
            assert first is not None and first["amount"] == 100

        # Failing transaction must roll back both DML and DDL cleanly.
        with pytest.raises(ValueError, match="abort tx"):
            async with db.transaction() as tx:
                await tx.execute("INSERT INTO ledger (amount) VALUES (?)", (999,))
                await tx.execute("CREATE TABLE rolled_back (id INTEGER PRIMARY KEY)")
                raise ValueError("abort tx")

        rows = await db.fetchall("SELECT amount FROM ledger ORDER BY id")
        assert [r["amount"] for r in rows] == [100, 200]
        rb_table = await db.fetchone(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'rolled_back'"
        )
        assert rb_table is None

    # Verify committed rows persisted on disk across reconnect.
    async with Database(db_path) as db:
        rows = await db.fetchall("SELECT amount FROM ledger ORDER BY id")
        assert [r["amount"] for r in rows] == [100, 200]


async def test_transaction_rollback_on_cancellation(tmp_path: Path) -> None:
    """CancelledError inside transaction() rolls back and leaves DB healthy."""
    async with Database(tmp_path / "tx_cancel.db") as db:
        await db.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT)")
        entered = asyncio.Event()

        async def worker() -> None:
            async with db.transaction() as tx:
                await tx.execute("INSERT INTO kv (k, v) VALUES (?, ?)", ("a", "1"))
                entered.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Uncommitted write was rolled back and DB accepts new transactions.
        assert await db.fetchall("SELECT * FROM kv") == []
        async with db.transaction() as tx:
            await tx.execute("INSERT INTO kv (k, v) VALUES (?, ?)", ("b", "2"))
        row = await db.fetchone("SELECT v FROM kv WHERE k = ?", ("b",))
        assert row is not None and row["v"] == "2"


async def test_reentrant_transaction_and_stale_tx_handle_errors(tmp_path: Path) -> None:
    """Re-entrant write calls inside transaction() and stale tx handles raise."""
    async with Database(tmp_path / "reentrant.db") as db:
        await db.execute("CREATE TABLE nums (n INTEGER)")

        async with db.transaction() as tx:
            await tx.execute("INSERT INTO nums (n) VALUES (?)", (1,))

            with pytest.raises(RuntimeError, match="Nested transactions"):
                async with db.transaction():
                    pass

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.execute("INSERT INTO nums (n) VALUES (?)", (2,))

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.migrate("m", ["CREATE TABLE m (id INTEGER PRIMARY KEY)"])

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.close()

            # Outer transaction still commits normally after catching errors.
            await tx.execute("INSERT INTO nums (n) VALUES (?)", (3,))

        rows = await db.fetchall("SELECT n FROM nums ORDER BY n")
        assert [r["n"] for r in rows] == [1, 3]

        # Using tx handle after exiting context must fail.
        with pytest.raises(RuntimeError, match="no longer active"):
            await tx.execute("INSERT INTO nums (n) VALUES (?)", (4,))
        with pytest.raises(RuntimeError, match="no longer active"):
            await tx.fetchone("SELECT 1")
        with pytest.raises(RuntimeError, match="no longer active"):
            await tx.fetchall("SELECT 1")


async def test_migrate_incremental_multi_module_and_multi_statement(
    tmp_path: Path,
) -> None:
    """migrate() tracks independent module versions and supports multi-statement SQL."""
    async with Database(tmp_path / "migrate.db") as db:
        alpha_v1 = [
            """
            -- Create alpha table with semicolon in default literal and trigger
            CREATE TABLE alpha (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT NOT NULL DEFAULT 'a;b'
            );;
            CREATE TABLE alpha_audit (alpha_id INTEGER NOT NULL);
            CREATE TRIGGER trg_alpha AFTER INSERT ON alpha
            BEGIN
                INSERT INTO alpha_audit (alpha_id) VALUES (NEW.id);
            END;
            CREATE INDEX idx_alpha_tag ON alpha (tag);
            """
        ]
        beta_v1 = ["CREATE TABLE beta (k TEXT PRIMARY KEY, v INTEGER NOT NULL)"]

        await db.migrate("alpha", alpha_v1)
        await db.migrate("beta", beta_v1)

        # Re-running same steps is idempotent.
        await db.migrate("alpha", alpha_v1)

        # Trigger & default literal from multi-statement migration work.
        res = await db.execute("INSERT INTO alpha DEFAULT VALUES")
        row = await db.fetchone("SELECT tag FROM alpha WHERE id = ?", (res.lastrowid,))
        audit = await db.fetchone("SELECT alpha_id FROM alpha_audit")
        assert row is not None and row["tag"] == "a;b"
        assert audit is not None and audit["alpha_id"] == res.lastrowid

        # Upgrade alpha to v2 without affecting beta.
        alpha_v2 = [*alpha_v1, "ALTER TABLE alpha ADD COLUMN extra TEXT DEFAULT ''"]
        await db.migrate("alpha", alpha_v2)

        versions = await db.fetchall(
            "SELECT module, version FROM _migrations ORDER BY module"
        )
        assert [(r["module"], r["version"]) for r in versions] == [
            ("alpha", 2),
            ("beta", 1),
        ]

        # Verify column from v2 exists.
        await db.execute("INSERT INTO alpha (tag, extra) VALUES (?, ?)", ("x", "ok"))
        added = await db.fetchone("SELECT extra FROM alpha WHERE tag = ?", ("x",))
        assert added is not None and added["extra"] == "ok"


async def test_migrate_rollback_on_bad_sql_and_validation_errors(
    tmp_path: Path,
) -> None:
    """Failed migration step rolls back atomically; shrunk/invalid args raise."""
    async with Database(tmp_path / "migrate_err.db") as db:
        step1 = "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)"
        bad_step2 = (
            "CREATE TABLE temp_should_rollback (id INTEGER PRIMARY KEY); "
            "THIS IS NOT VALID SQL;"
        )

        with pytest.raises(aiosqlite.OperationalError):
            await db.migrate("notes", [step1, bad_step2])

        # Step 1 committed, step 2 rolled back completely (including its first DDL).
        ver = await db.fetchone(
            "SELECT version FROM _migrations WHERE module = ?",
            ("notes",),
        )
        assert ver is not None and ver["version"] == 1
        assert (
            await db.fetchone(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'notes'"
            )
            is not None
        )
        assert (
            await db.fetchone(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'temp_should_rollback'"
            )
            is None
        )

        # Shrinking migration history must raise ValueError.
        with pytest.raises(ValueError, match="shrunk"):
            await db.migrate("notes", [])

        # Empty module name, empty step, or bare string steps must raise.
        with pytest.raises(ValueError, match="non-empty"):
            await db.migrate("", [step1])
        with pytest.raises(ValueError, match="non-empty"):
            await db.migrate("   ", [step1])
        with pytest.raises(TypeError, match="not a single str"):
            await db.migrate("notes", step1)
        with pytest.raises(ValueError, match="empty"):
            await db.migrate("empty_step_mod", ["   ;  "])


async def test_concurrent_reads_writes_and_transactions(tmp_path: Path) -> None:
    """Concurrent fetchall, execute, and transaction tasks run without errors."""
    async with Database(tmp_path / "concurrency.db") as db:
        await db.execute(
            "CREATE TABLE counters (id INTEGER PRIMARY KEY AUTOINCREMENT, val INTEGER)"
        )

        async def writer(value: int) -> None:
            await db.execute("INSERT INTO counters (val) VALUES (?)", (value,))

        async def tx_writer(value: int) -> None:
            async with db.transaction() as tx:
                await tx.execute("INSERT INTO counters (val) VALUES (?)", (value,))
                await asyncio.sleep(0)
                await tx.execute("INSERT INTO counters (val) VALUES (?)", (value * 10,))

        async def reader() -> int:
            rows = await db.fetchall("SELECT val FROM counters")
            return len(rows)

        tasks: list[asyncio.Future[object] | asyncio.Task[object]] = []
        for i in range(1, 21):
            tasks.append(asyncio.create_task(writer(i)))
            tasks.append(asyncio.create_task(reader()))
        for i in range(1, 11):
            tasks.append(asyncio.create_task(tx_writer(100 + i)))
            tasks.append(asyncio.create_task(reader()))

        await asyncio.gather(*tasks)

        total = await db.fetchone("SELECT COUNT(*) AS cnt FROM counters")
        assert total is not None and total["cnt"] == 40
