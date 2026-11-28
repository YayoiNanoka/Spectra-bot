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
        "idx_messages_platform_id",
    }.issubset(names)

    mpi_indexes = await db.fetchall(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'index' AND tbl_name = 'message_platform_ids'"
    )
    assert "idx_message_platform_ids_lookup" in {row["name"] for row in mpi_indexes}

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

    with pytest.raises(TypeError, match="sequence"):
        await store.append_message(
            group_id="1",
            user_id="u1",
            sender_name="U1",
            role="user",
            content="bad extra",
            extra_platform_ids="single-str",  # type: ignore[arg-type]
        )


async def test_platform_message_ids_resolution_scoping_and_cascade(
    store: MessageStore,
) -> None:
    # Whitespace IDs normalize to None.
    blank_id = await store.append_message(
        group_id="1001",
        user_id="u1",
        sender_name="Alice",
        role="user",
        content="plain",
        created_at=100,
        platform_message_id="   ",
        reply_to_message_id="  ",
        extra_platform_ids=("", "  "),
    )
    rows = await store.get_group_messages("1001")
    assert rows[0].id == blank_id
    assert rows[0].platform_message_id is None
    assert rows[0].reply_to_message_id is None

    # Segmented assistant reply with multiple bubbles + duplicate stub IDs.
    bot_msg_id = await store.append_message(
        group_id="1001",
        user_id="99999",
        sender_name="Spectra",
        role="assistant",
        content="第一句。第二句！第三句？",
        created_at=200,
        platform_message_id="b1",
        extra_platform_ids=["b1", "b2", "b3", "b2", "  "],
    )
    # Same platform_message_id in another group must stay isolated.
    other_group_msg_id = await store.append_message(
        group_id="2002",
        user_id="99999",
        sender_name="Spectra",
        role="assistant",
        content="other group",
        created_at=210,
        platform_message_id="b2",
    )
    # Private chat with user u1.
    dm_msg_id = await store.append_message(
        group_id=None,
        user_id="u1",
        sender_name="Spectra",
        role="assistant",
        content="dm multi",
        created_at=220,
        platform_message_id="dm-1",
        extra_platform_ids=["dm-1", "dm-2"],
    )
    # Later message reusing a platform ID in same group resolves to latest id.
    newer_dup_id = await store.append_message(
        group_id="1001",
        user_id="u2",
        sender_name="Bob",
        role="user",
        content="reused id",
        created_at=230,
        platform_message_id="b3",
        reply_to_message_id="b2",
    )

    assert (
        await store.resolve_platform_message_id(
            platform_message_id="b1", group_id="1001", user_id="u1"
        )
        == bot_msg_id
    )
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="b2", group_id="1001", user_id="u1"
        )
        == bot_msg_id
    )
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="b3", group_id="1001", user_id="u1"
        )
        == newer_dup_id
    )
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="b2", group_id="2002", user_id="u1"
        )
        == other_group_msg_id
    )
    # Cross-chat isolation: group ID not visible in private chat and vice versa.
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="b2", group_id=None, user_id="u1"
        )
        is None
    )
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="dm-2", group_id=None, user_id="u1"
        )
        == dm_msg_id
    )
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="dm-2", group_id=None, user_id="u2"
        )
        is None
    )
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="   ", group_id="1001", user_id="u1"
        )
        is None
    )

    # Deleting messages row cascades to message_platform_ids.
    deleted = await store.cleanup_expired_messages(before_created_at=205)
    assert deleted == 2
    assert (
        await store.resolve_platform_message_id(
            platform_message_id="b2", group_id="1001", user_id="u1"
        )
        is None
    )
    remaining_mpi = await store._db.fetchall(
        "SELECT message_id, platform_message_id FROM message_platform_ids "
        "WHERE message_id = ?",
        (bot_msg_id,),
    )
    assert remaining_mpi == []


async def test_get_messages_in_range(store: MessageStore) -> None:
    g_ids: list[int] = []
    for idx in range(1, 5):
        g_ids.append(
            await store.append_message(
                group_id="1001",
                user_id="u1",
                sender_name="Alice",
                role="user",
                content=f"g-{idx}",
                created_at=100 + idx,
            )
        )
    p_id = await store.append_message(
        group_id=None,
        user_id="u1",
        sender_name="Alice",
        role="user",
        content="dm-1",
        created_at=105,
    )
    g5_id = await store.append_message(
        group_id="1001",
        user_id="u2",
        sender_name="Bob",
        role="user",
        content="g-5",
        created_at=106,
    )

    sliced = await store.get_messages_in_range(
        group_id="1001",
        user_id="u1",
        start_id=g_ids[1],
        end_id=g5_id,
    )
    assert [m.id for m in sliced] == [g_ids[1], g_ids[2], g_ids[3], g5_id]
    assert [m.content for m in sliced] == ["g-2", "g-3", "g-4", "g-5"]

    dm_sliced = await store.get_messages_in_range(
        group_id=None,
        user_id="u1",
        start_id=g_ids[0],
        end_id=g5_id,
    )
    assert [m.id for m in dm_sliced] == [p_id]

    assert (
        await store.get_messages_in_range(
            group_id="1001",
            user_id="u1",
            start_id=g5_id,
            end_id=g_ids[0],
        )
        == []
    )
