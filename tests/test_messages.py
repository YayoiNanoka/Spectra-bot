"""Unit tests for spectra.messages."""

from __future__ import annotations

import pytest

from spectra.db import Database
from spectra.messages import MessageStore


@pytest.fixture
async def store() -> MessageStore:
    db = Database(":memory:")
    await db.connect()
    msg_store = MessageStore(db)
    await msg_store.init()
    try:
        yield msg_store
    finally:
        await db.close()


async def test_migration_creates_table_and_three_indexes(store: MessageStore) -> None:
    db = store._db
    rows = await db.fetchall(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'messages'"
    )
    names = {row["name"] for row in rows}
    assert {
        "idx_messages_group_id_id",
        "idx_messages_private_user_id",
        "idx_messages_created_at",
    }.issubset(names)

    # Idempotent re-init.
    await store.init()


async def test_group_messages_chronological_and_cursor(store: MessageStore) -> None:
    ids = []
    for idx in range(1, 6):
        msg_id = await store.append_message(
            group_id="1001",
            user_id=f"u{idx}",
            sender_name=f"User {idx}",
            role="user" if idx % 2 else "assistant",
            content=f"msg-{idx}",
            created_at=1000 + idx,
        )
        ids.append(msg_id)

    # Unrelated group & private rows must not leak into group 1001 query.
    await store.append_message(
        group_id="2002",
        user_id="u9",
        sender_name="Other",
        role="user",
        content="other-group",
        created_at=2000,
    )
    await store.append_message(
        group_id=None,
        user_id="u1",
        sender_name="User 1",
        role="user",
        content="private-msg",
        created_at=2001,
    )

    latest = await store.get_group_messages("1001", limit=3)
    assert [m.id for m in latest] == ids[-3:]
    assert [m.content for m in latest] == ["msg-3", "msg-4", "msg-5"]

    after = await store.get_group_messages("1001", limit=2, after_id=ids[1])
    assert [m.id for m in after] == [ids[2], ids[3]]
    assert [m.content for m in after] == ["msg-3", "msg-4"]

    assert await store.get_group_messages("1001", limit=0) == []


async def test_private_messages_interleave_user_and_assistant(
    store: MessageStore,
) -> None:
    id1 = await store.append_message(
        group_id=None,
        user_id="42",
        sender_name="Alice",
        role="user",
        content="hi",
        created_at=100,
    )
    id2 = await store.append_message(
        group_id=None,
        user_id="42",
        sender_name="Spectra",
        role="assistant",
        content="hello!",
        created_at=101,
    )
    id3 = await store.append_message(
        group_id=None,
        user_id="42",
        sender_name="Alice",
        role="user",
        content="how are you?",
        created_at=102,
    )
    # Same user_id in a group must NOT appear in private messages.
    await store.append_message(
        group_id="777",
        user_id="42",
        sender_name="Alice",
        role="user",
        content="in group",
        created_at=103,
    )
    # Another private user must NOT appear.
    await store.append_message(
        group_id=None,
        user_id="99",
        sender_name="Bob",
        role="user",
        content="other private",
        created_at=104,
    )

    latest = await store.get_private_messages("42", limit=10)
    assert [(m.id, m.role, m.sender_name, m.content) for m in latest] == [
        (id1, "user", "Alice", "hi"),
        (id2, "assistant", "Spectra", "hello!"),
        (id3, "user", "Alice", "how are you?"),
    ]

    latest_two = await store.get_private_messages("42", limit=2)
    assert [m.id for m in latest_two] == [id2, id3]

    after_first = await store.get_private_messages("42", limit=10, after_id=id1)
    assert [m.id for m in after_first] == [id2, id3]


async def test_cleanup_expired_messages(store: MessageStore) -> None:
    id1 = await store.append_message(
        group_id="1",
        user_id="u1",
        sender_name="U1",
        role="user",
        content="old-1",
        created_at=100,
    )
    id2 = await store.append_message(
        group_id="1",
        user_id="u1",
        sender_name="U1",
        role="user",
        content="old-2",
        created_at=200,
    )
    id3 = await store.append_message(
        group_id="1",
        user_id="u1",
        sender_name="U1",
        role="user",
        content="new-3",
        created_at=500,
    )

    # Protect id >= id2 with max_deletable_id=id2.
    deleted = await store.cleanup_expired_messages(
        before_created_at=300,
        max_deletable_id=id2,
    )
    assert deleted == 1
    remaining = await store.get_group_messages("1", limit=10)
    assert [m.id for m in remaining] == [id2, id3]
    assert id1 not in [m.id for m in remaining]

    # Without max_deletable_id, delete all rows older than 300.
    deleted_all_old = await store.cleanup_expired_messages(before_created_at=300)
    assert deleted_all_old == 1
    remaining_after = await store.get_group_messages("1", limit=10)
    assert [m.id for m in remaining_after] == [id3]


async def test_invalid_role_and_negative_limit(store: MessageStore) -> None:
    with pytest.raises(ValueError, match="Invalid message role"):
        await store.append_message(
            group_id="1",
            user_id="u1",
            sender_name="U1",
            role="system",  # type: ignore[arg-type]
            content="bad",
        )

    with pytest.raises(ValueError, match="non-negative"):
        await store.get_group_messages("1", limit=-1)

    with pytest.raises(ValueError, match="non-negative"):
        await store.get_private_messages("u1", limit=-1)
