"""Unit tests for spectra.db generic async SQLite base."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path

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

        class _CancellingCtx:
            async def __aenter__(self) -> object:
                raise asyncio.CancelledError()

            async def __aexit__(self, *args: object) -> None:
                return None

        def cancelling_execute(*_args: object, **_kwargs: object) -> _CancellingCtx:
            return _CancellingCtx()

        conn.execute = cancelling_execute  # type: ignore[method-assign]
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


async def test_transaction_commit_fails_when_implicitly_aborted(tmp_path: Path) -> None:
    """Unconditional COMMIT fails loudly if SQLite aborted the tx mid-block."""
    async with Database(tmp_path / "implicit_abort.db") as db:
        await db.execute("CREATE TABLE uniq (id INTEGER PRIMARY KEY)")
        await db.execute("INSERT INTO uniq (id) VALUES (1)")

        with pytest.raises(aiosqlite.OperationalError, match="no transaction"):
            async with db.transaction() as tx:
                await tx.execute("INSERT INTO uniq (id) VALUES (2)")
                with pytest.raises(aiosqlite.IntegrityError):
                    await tx.execute("INSERT OR ROLLBACK INTO uniq (id) VALUES (1)")

        # Connection stays usable and row 2 was rolled back.
        rows = await db.fetchall("SELECT id FROM uniq ORDER BY id")
        assert [r["id"] for r in rows] == [1]


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


async def test_cancelled_during_begin_immediate_in_transaction_and_migrate(
    tmp_path: Path,
) -> None:
    """Cancellation after BEGIN IMMEDIATE is enqueued still rolls back cleanly."""
    async with Database(tmp_path / "cancel_begin.db") as db:
        assert db._conn is not None
        conn = db._conn
        original_execute = conn.execute
        cancel_next_begin = False

        class _CancelAfterEnqueueBegin:
            def __init__(self, inner: object) -> None:
                self._inner = inner

            async def __aenter__(self) -> object:
                # Execute real BEGIN IMMEDIATE on the worker thread, then simulate
                # CancelledError arriving while awaiting its completion.
                await self._inner.__aenter__()  # type: ignore[attr-defined]
                await self._inner.__aexit__(None, None, None)  # type: ignore[attr-defined]
                raise asyncio.CancelledError()

            async def __aexit__(self, *args: object) -> None:
                return None

        def hooked_execute(sql: str, *args: object, **kwargs: object):
            nonlocal cancel_next_begin
            if cancel_next_begin and sql == "BEGIN IMMEDIATE":
                cancel_next_begin = False
                return _CancelAfterEnqueueBegin(original_execute(sql, *args, **kwargs))
            return original_execute(sql, *args, **kwargs)

        conn.execute = hooked_execute  # type: ignore[method-assign]

        # 1) Cancelled during transaction()'s BEGIN IMMEDIATE.
        cancel_next_begin = True
        with pytest.raises(asyncio.CancelledError):
            async with db.transaction():
                pytest.fail("body should not execute")

        # Subsequent transaction must succeed (connection is NOT stuck in a tx).
        async with db.transaction() as tx:
            await tx.execute("CREATE TABLE ok (id INTEGER PRIMARY KEY)")

        # 2) Cancelled during migrate()'s BEGIN IMMEDIATE.
        cancel_next_begin = True
        with pytest.raises(asyncio.CancelledError):
            await db.migrate("m_cancel", ["CREATE TABLE m1 (id INTEGER PRIMARY KEY)"])

        # Subsequent migrate must succeed cleanly.
        await db.migrate("m_cancel", ["CREATE TABLE m1 (id INTEGER PRIMARY KEY)"])
        ver = await db.fetchone(
            "SELECT version FROM _migrations WHERE module = ?",
            ("m_cancel",),
        )
        assert ver is not None and ver["version"] == 1


async def test_reentrant_calls_and_child_tasks_inside_transaction_raise(
    tmp_path: Path,
) -> None:
    """Re-entrant DB calls, child tasks, and stale tx handles raise immediately."""
    async with (
        Database(tmp_path / "reentrant.db") as db,
        Database(tmp_path / "other.db") as other_db,
    ):
        await db.execute("CREATE TABLE nums (n INTEGER)")
        await other_db.execute("CREATE TABLE other (n INTEGER)")

        async with db.transaction() as tx:
            await tx.execute("INSERT INTO nums (n) VALUES (?)", (1,))

            with pytest.raises(RuntimeError, match="Nested transactions"):
                async with db.transaction():
                    pass

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.execute("INSERT INTO nums (n) VALUES (?)", (2,))

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.fetchone("SELECT 1")

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.fetchall("SELECT 1")

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.migrate("m", ["CREATE TABLE m (id INTEGER PRIMARY KEY)"])

            with pytest.raises(RuntimeError, match="active transaction"):
                await db.close()

            # Child tasks spawned via wait_for / TaskGroup inherit ContextVar and
            # fail fast instead of deadlocking on the exclusive guard.
            with pytest.raises(RuntimeError, match="active transaction"):
                await asyncio.wait_for(
                    db.execute("INSERT INTO nums (n) VALUES (?)", (2,)),
                    timeout=1.0,
                )

            with pytest.raises(ExceptionGroup) as eg_info:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(db.fetchone("SELECT 1"))
            assert len(eg_info.value.exceptions) == 1
            assert isinstance(eg_info.value.exceptions[0], RuntimeError)

            # A distinct Database instance is unaffected by db's active transaction.
            async with other_db.transaction() as other_tx:
                await other_tx.execute("INSERT INTO other (n) VALUES (?)", (42,))

            # Outer transaction still commits normally after catching errors.
            await tx.execute("INSERT INTO nums (n) VALUES (?)", (3,))

        rows = await db.fetchall("SELECT n FROM nums ORDER BY n")
        assert [r["n"] for r in rows] == [1, 3]
        other_rows = await other_db.fetchall("SELECT n FROM other")
        assert [r["n"] for r in other_rows] == [42]

        # Using tx handle after exiting context must fail.
        with pytest.raises(RuntimeError, match="no longer active"):
            await tx.execute("INSERT INTO nums (n) VALUES (?)", (4,))
        with pytest.raises(RuntimeError, match="no longer active"):
            await tx.fetchone("SELECT 1")
        with pytest.raises(RuntimeError, match="no longer active"):
            await tx.fetchall("SELECT 1")


async def test_concurrent_reads_isolated_from_tx_rollback_and_drop_table(
    tmp_path: Path,
) -> None:
    """Readers never observe uncommitted rows, rollback errors, or table lock errors."""
    async with Database(tmp_path / "isolation.db") as db:
        await db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
        await db.execute("INSERT INTO items (id, val) VALUES (1, 'committed')")

        tx_started = asyncio.Event()
        dirty_values: list[str] = []

        async def rolling_back_tx() -> None:
            with pytest.raises(ValueError, match="rollback now"):
                async with db.transaction() as tx:
                    await tx.execute(
                        "INSERT INTO items (id, val) VALUES (2, 'uncommitted')"
                    )
                    tx_started.set()
                    await asyncio.sleep(0.03)
                    raise ValueError("rollback now")

        async def reader_loop() -> None:
            await tx_started.wait()
            for _ in range(10):
                rows = await db.fetchall("SELECT val FROM items")
                for row in rows:
                    if row["val"] == "uncommitted":
                        dirty_values.append(row["val"])
                one = await db.fetchone("SELECT val FROM items WHERE id = 2")
                if one is not None:
                    dirty_values.append(one["val"])

        await asyncio.gather(rolling_back_tx(), reader_loop(), reader_loop())
        assert dirty_values == []

        # Concurrent read + DROP TABLE must not trigger 'database table is locked'.
        await db.execute("CREATE TABLE ephemeral (id INTEGER PRIMARY KEY)")
        await db.execute("INSERT INTO ephemeral (id) VALUES (1)")

        assert db._conn is not None
        conn = db._conn
        original_execute = conn.execute
        read_in_flight = asyncio.Event()
        release_read = asyncio.Event()

        class _PausedCursorCtx:
            def __init__(self, inner: object) -> None:
                self._inner = inner

            async def __aenter__(self) -> object:
                cursor = await self._inner.__aenter__()  # type: ignore[attr-defined]
                read_in_flight.set()
                await release_read.wait()
                return cursor

            async def __aexit__(self, *args: object) -> object:
                return await self._inner.__aexit__(*args)  # type: ignore[attr-defined]

        paused_once = False

        def slow_select(sql: str, *args: object, **kwargs: object):
            nonlocal paused_once
            if not paused_once and "FROM ephemeral" in sql:
                paused_once = True
                return _PausedCursorCtx(original_execute(sql, *args, **kwargs))
            return original_execute(sql, *args, **kwargs)

        conn.execute = slow_select  # type: ignore[method-assign]

        read_task = asyncio.create_task(db.fetchall("SELECT id FROM ephemeral"))
        await read_in_flight.wait()
        drop_task = asyncio.create_task(db.execute("DROP TABLE ephemeral"))
        await asyncio.sleep(0.01)
        assert not drop_task.done()
        release_read.set()

        read_rows, _ = await asyncio.gather(read_task, drop_task)
        assert [r["id"] for r in read_rows] == [1]


async def test_concurrent_close_waits_for_teardown_and_drains_readers(
    tmp_path: Path,
) -> None:
    """Concurrent close() calls wait until teardown completes and drain readers."""
    db = Database(tmp_path / "close_race.db")
    await db.connect()
    await db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")

    assert db._conn is not None
    conn = db._conn
    original_close = conn.close
    teardown_finished = False

    async def slow_close() -> None:
        nonlocal teardown_finished
        await asyncio.sleep(0.04)
        await original_close()
        teardown_finished = True

    conn.close = slow_close  # type: ignore[method-assign]

    async def close_and_verify() -> bool:
        await db.close()
        return teardown_finished

    results = await asyncio.gather(close_and_verify(), close_and_verify())
    assert results == [True, True]
    assert db._conn is None

    # Verify close() drains an in-flight read cleanly without closing early.
    await db.connect()
    await db.execute("INSERT INTO t (id) VALUES (7)")
    assert db._conn is not None
    conn2 = db._conn
    orig_exec2 = conn2.execute
    in_read = asyncio.Event()
    resume_read = asyncio.Event()

    class _SlowReadCtx:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        async def __aenter__(self) -> object:
            cursor = await self._inner.__aenter__()  # type: ignore[attr-defined]
            in_read.set()
            await resume_read.wait()
            return cursor

        async def __aexit__(self, *args: object) -> object:
            return await self._inner.__aexit__(*args)  # type: ignore[attr-defined]

    def hooked_exec2(sql: str, *args: object, **kwargs: object):
        if "FROM t" in sql:
            return _SlowReadCtx(orig_exec2(sql, *args, **kwargs))
        return orig_exec2(sql, *args, **kwargs)

    conn2.execute = hooked_exec2  # type: ignore[method-assign]

    read_task = asyncio.create_task(db.fetchall("SELECT id FROM t"))
    await in_read.wait()
    close_task = asyncio.create_task(db.close())
    await asyncio.sleep(0.01)
    assert not close_task.done()
    resume_read.set()
    rows, _ = await asyncio.gather(read_task, close_task)
    assert [r["id"] for r in rows] == [7]
    assert db._conn is None


async def test_migrate_incremental_multi_module_and_multi_statement(
    tmp_path: Path,
) -> None:
    """migrate() tracks independent module versions and supports multi-statement SQL."""
    async with Database(tmp_path / "migrate.db") as db:
        alpha_v1 = [
            """
            -- Create alpha table with ; in comment and default literal
            CREATE TABLE alpha (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT NOT NULL DEFAULT 'a;b'
            );;
            /* block comment with ; */ ;
            CREATE TABLE alpha_audit (alpha_id INTEGER NOT NULL);
            CREATE TRIGGER trg_alpha AFTER INSERT ON alpha
            BEGIN
                INSERT INTO alpha_audit (alpha_id) VALUES (NEW.id);
            END;
            CREATE INDEX idx_alpha_tag ON alpha (tag);
            -- trailing comment after last semicolon;
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
    """Failed migration rolls back atomically; comments/manual tx/shrunk raise."""
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

        # Unclosed block comment fails validation before running any DDL.
        with pytest.raises(ValueError, match="Unterminated block comment"):
            await db.migrate(
                "unclosed_mod",
                ["CREATE TABLE should_not_exist (id INTEGER PRIMARY KEY); /* unclosed"],
            )
        with pytest.raises(ValueError, match="Unterminated block comment"):
            await db.migrate("unclosed_mod", ["/* unclosed"])
        assert (
            await db.fetchone(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'should_not_exist'"
            )
            is None
        )

        # Shrinking migration history must raise ValueError.
        with pytest.raises(ValueError, match="shrunk"):
            await db.migrate("notes", [])

        # Empty module name, empty/comment-only step, or bare string steps must raise.
        with pytest.raises(ValueError, match="non-empty"):
            await db.migrate("", [step1])
        with pytest.raises(ValueError, match="non-empty"):
            await db.migrate("   ", [step1])
        with pytest.raises(TypeError, match="not a single str"):
            await db.migrate("notes", step1)
        for empty_step in ("   ;  ", "-- todo", "/* note */;", "  -- a\n/* b */ ; "):
            with pytest.raises(ValueError, match="empty"):
                await db.migrate("empty_step_mod", [empty_step])

        # Manual transaction control statements must be rejected before running DDL.
        for bad_tx_step in (
            "CREATE TABLE leaked (id INTEGER PRIMARY KEY); COMMIT; INVALID SQL;",
            "-- comment\nBEGIN; CREATE TABLE leaked (id INTEGER PRIMARY KEY);",
            "/* comment */ ROLLBACK;",
            "SAVEPOINT sp1; RELEASE sp1;",
            "END;",
        ):
            with pytest.raises(ValueError, match="manual transaction control"):
                await db.migrate("manual_tx_mod", [bad_tx_step])

        assert (
            await db.fetchone(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'leaked'"
            )
            is None
        )
        assert (
            await db.fetchone(
                "SELECT version FROM _migrations WHERE module = ?",
                ("manual_tx_mod",),
            )
            is None
        )


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
