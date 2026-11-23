"""Generic async SQLite database base for Spectra-bot.

This module intentionally contains no business-domain tables. Feature modules
register and evolve their own schemas independently via :meth:`Database.migrate`.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiosqlite

logger = logging.getLogger(__name__)

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=30000",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA temp_store=MEMORY",
)

_CREATE_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS _migrations (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL
)
"""

_UPSERT_MIGRATION_SQL = """
INSERT INTO _migrations (module, version)
VALUES (?, ?)
ON CONFLICT(module) DO UPDATE SET version = excluded.version
"""


@dataclass(slots=True, frozen=True)
class ExecuteResult:
    """Metadata returned by a write SQL statement.

    Attributes:
        lastrowid: Row ID of the last inserted row, or ``0`` if unavailable.
        rowcount: Number of rows affected by the statement (or ``-1`` when
            SQLite does not report a row count).
    """

    lastrowid: int
    rowcount: int


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL string into individual executable statements.

    Uses :func:`sqlite3.complete_statement` so semicolons inside string
    literals, comments, or ``CREATE TRIGGER ... BEGIN ... END`` blocks do not
    split a statement prematurely.

    Args:
        sql: Raw SQL text containing one or more statements.

    Returns:
        A list of stripped, non-empty SQL statements in execution order.
    """
    statements: list[str] = []
    buf = ""
    pos = 0
    while True:
        semi = sql.find(";", pos)
        if semi == -1:
            tail = (buf + sql[pos:]).strip()
            if tail:
                statements.append(tail)
            break
        buf += sql[pos : semi + 1]
        pos = semi + 1
        if buf.strip() == ";":
            buf = ""
            continue
        if sqlite3.complete_statement(buf):
            stmt = buf.strip()
            if stmt:
                statements.append(stmt)
            buf = ""
    return statements


class Transaction:
    """Active SQLite transaction handle yielded by :meth:`Database.transaction`.

    All operations execute on the underlying connection while the outer
    :class:`Database` write lock is already held.
    """

    __slots__ = ("_conn", "_active")

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        self._active = True

    def _ensure_active(self) -> aiosqlite.Connection:
        if not self._active:
            raise RuntimeError("Transaction is no longer active.")
        return self._conn

    async def fetchone(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> aiosqlite.Row | None:
        """Execute a read query and return the first row, or ``None``.

        Args:
            sql: SQL query string.
            params: Positional query parameters.

        Returns:
            The first matching :class:`aiosqlite.Row`, or ``None``.

        Raises:
            RuntimeError: If the transaction has already exited.
        """
        conn = self._ensure_active()
        async with conn.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def fetchall(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> list[aiosqlite.Row]:
        """Execute a read query and return all matching rows.

        Args:
            sql: SQL query string.
            params: Positional query parameters.

        Returns:
            List of matching :class:`aiosqlite.Row` objects.

        Raises:
            RuntimeError: If the transaction has already exited.
        """
        conn = self._ensure_active()
        async with conn.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    async def execute(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> ExecuteResult:
        """Execute a single write statement inside the active transaction.

        Args:
            sql: Single SQL statement.
            params: Positional statement parameters.

        Returns:
            :class:`ExecuteResult` containing ``lastrowid`` and ``rowcount``.

        Raises:
            RuntimeError: If the transaction has already exited.
        """
        conn = self._ensure_active()
        async with conn.execute(sql, params) as cursor:
            return ExecuteResult(
                lastrowid=cursor.lastrowid or 0,
                rowcount=cursor.rowcount,
            )


class Database:
    """Async SQLite connection wrapper with write serialization and migrations.

    Uses ``isolation_level=None`` (explicit transaction control) so DDL and DML
    statements participate cleanly in atomic ``BEGIN IMMEDIATE`` transactions.
    Writes are serialized with an :class:`asyncio.Lock`, while reads
    (:meth:`fetchone` / :meth:`fetchall`) execute without acquiring the write
    lock.

    Args:
        db_path: Path to the SQLite database file, or ``":memory:"`` for an
            in-memory database.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._is_memory = str(db_path) == ":memory:"
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._tx_owner: asyncio.Task[Any] | None = None

    @property
    def db_path(self) -> Path:
        """Return the configured database path."""
        return self._db_path

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self._conn

    def _ensure_not_in_tx(self, message: str) -> None:
        if self._tx_owner is not None and self._tx_owner is asyncio.current_task():
            raise RuntimeError(message)

    async def connect(self) -> None:
        """Open the SQLite connection, apply PRAGMAs, and ensure ``_migrations``.

        Safe to call if already connected (no-op). If initialization fails
        partway through, the underlying connection is closed cleanly before
        re-raising so no worker thread is leaked.
        """
        if self._conn is not None:
            return
        async with self._write_lock:
            if self._conn is not None:
                return
            if not self._is_memory:
                await asyncio.to_thread(
                    self._db_path.parent.mkdir,
                    parents=True,
                    exist_ok=True,
                )
            conn = await aiosqlite.connect(str(self._db_path), isolation_level=None)
            try:
                conn.row_factory = aiosqlite.Row
                for pragma in _PRAGMAS:
                    await conn.execute(pragma)
                await conn.execute(_CREATE_MIGRATIONS_TABLE_SQL)
            except BaseException:
                await conn.close()
                raise
            self._conn = conn

    async def close(self) -> None:
        """Close the SQLite connection.

        Idempotent: calling :meth:`close` when already closed is a no-op.

        Raises:
            RuntimeError: If called from inside an active :meth:`transaction`
                on the same task.
        """
        if self._conn is None:
            return
        self._ensure_not_in_tx(
            "Cannot close database while inside an active transaction."
        )
        async with self._write_lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                await conn.close()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def fetchone(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> aiosqlite.Row | None:
        """Execute a lock-free read query and return the first row, or ``None``.

        Args:
            sql: SQL query string.
            params: Positional query parameters.

        Returns:
            The first matching :class:`aiosqlite.Row`, or ``None``.

        Raises:
            RuntimeError: If the database is not connected.
        """
        conn = self._require_conn()
        async with conn.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def fetchall(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> list[aiosqlite.Row]:
        """Execute a lock-free read query and return all matching rows.

        Args:
            sql: SQL query string.
            params: Positional query parameters.

        Returns:
            List of matching :class:`aiosqlite.Row` objects.

        Raises:
            RuntimeError: If the database is not connected.
        """
        conn = self._require_conn()
        async with conn.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    async def execute(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> ExecuteResult:
        """Execute a single write statement under the write lock.

        Args:
            sql: Single SQL statement.
            params: Positional statement parameters.

        Returns:
            :class:`ExecuteResult` containing ``lastrowid`` and ``rowcount``.

        Raises:
            RuntimeError: If the database is not connected or called from
                inside an active :meth:`transaction` on the same task.
        """
        self._require_conn()
        self._ensure_not_in_tx(
            "Cannot call Database.execute() inside an active transaction; "
            "use the transaction handle instead."
        )
        async with self._write_lock:
            conn = self._require_conn()
            async with conn.execute(sql, params) as cursor:
                return ExecuteResult(
                    lastrowid=cursor.lastrowid or 0,
                    rowcount=cursor.rowcount,
                )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        """Run a block atomically inside ``BEGIN IMMEDIATE ... COMMIT``.

        Acquires the write lock, starts a ``BEGIN IMMEDIATE`` transaction, and
        yields a :class:`Transaction` handle. Commits on normal exit and rolls
        back on any ``BaseException`` (including :class:`asyncio.CancelledError`).

        Yields:
            An active :class:`Transaction` handle.

        Raises:
            RuntimeError: If the database is not connected or if nested
                re-entrantly on the same task.
        """
        self._require_conn()
        self._ensure_not_in_tx(
            "Nested transactions are not supported on the same Database instance."
        )
        async with self._write_lock:
            conn = self._require_conn()
            self._tx_owner = asyncio.current_task()
            try:
                await conn.execute("BEGIN IMMEDIATE")
                tx = Transaction(conn)
                try:
                    yield tx
                    if conn.in_transaction:
                        await conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        try:
                            await conn.execute("ROLLBACK")
                        except Exception:
                            logger.exception("Failed to roll back transaction")
                    raise
                finally:
                    tx._active = False
            finally:
                self._tx_owner = None

    async def migrate(self, module: str, steps: Sequence[str]) -> None:
        """Apply incremental schema migrations for a logical module.

        Each pending step ``steps[v]`` (from the module's recorded version to
        ``len(steps) - 1``) is executed atomically inside a
        ``BEGIN IMMEDIATE ... COMMIT`` transaction together with updating
        ``_migrations``. Multi-statement steps are split safely without
        calling ``executescript()``, preserving transaction atomicity. Steps
        must not contain manual transaction control statements.

        Args:
            module: Non-empty logical module name (e.g. ``"messages"``).
            steps: Ordered sequence of SQL migration steps. Index ``i``
                corresponds to target schema version ``i + 1``.

        Raises:
            RuntimeError: If the database is not connected or called inside an
                active :meth:`transaction` on the same task.
            TypeError: If ``steps`` is passed as a bare ``str``.
            ValueError: If ``module`` is empty, ``len(steps)`` is smaller than
                the recorded version, or a pending migration step is empty.
        """
        if not module or not module.strip():
            raise ValueError("Migration module name must be non-empty.")
        if isinstance(steps, str):
            raise TypeError(
                "Migration steps must be a sequence of SQL strings, not a single str."
            )
        self._require_conn()
        self._ensure_not_in_tx("Cannot call migrate() inside an active transaction.")

        async with self._write_lock:
            conn = self._require_conn()
            async with conn.execute(
                "SELECT version FROM _migrations WHERE module = ?",
                (module,),
            ) as cursor:
                row = await cursor.fetchone()
            current_version = int(row["version"]) if row is not None else 0

            if len(steps) < current_version:
                raise ValueError(
                    f"Migration history shrunk for module {module!r}: "
                    f"database is at version {current_version}, "
                    f"but only {len(steps)} step(s) were provided."
                )

            if current_version == len(steps):
                return

            for version in range(current_version + 1, len(steps) + 1):
                statements = _split_sql_statements(steps[version - 1])
                if not statements:
                    raise ValueError(
                        f"Migration step {version} for module {module!r} is empty."
                    )

                await conn.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in statements:
                        await conn.execute(stmt)
                    await conn.execute(_UPSERT_MIGRATION_SQL, (module, version))
                    if conn.in_transaction:
                        await conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        try:
                            await conn.execute("ROLLBACK")
                        except Exception:
                            logger.exception(
                                "Failed to roll back migration step %d for module %r",
                                version,
                                module,
                            )
                    raise
