"""Unit tests for spectra.memory (MemoryStore, privacy isolation, rerank, jargon)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import httpx
import pytest

from spectra.config import EmbeddingConfig, JevConfig, MemoryConfig
from spectra.db import _ACTIVE_TX_DBS, Database
from spectra.embedding import (
    EmbeddingClient,
    EmbeddingResult,
    format_embedding_model_tag,
    pack_normalized_vector,
)
from spectra.jev import JevClient
from spectra.memory import (
    MemoryStore,
    PublicUserProfile,
    UserProfile,
)


def _make_embedding(values: list[float], model: str = "test-embed") -> EmbeddingResult:
    blob, dim = pack_normalized_vector(values)
    return EmbeddingResult(
        blob=blob,
        dimension=dim,
        model_tag=format_embedding_model_tag(model, dim),
    )


async def test_memory_store_migrations_and_group_profile_crud() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(db, MemoryConfig())
        await store.init()
        await store.init()  # idempotent

        row = await db.fetchone(
            "SELECT version FROM _migrations WHERE module = 'memory'"
        )
        assert row is not None
        assert int(row["version"]) == 1

        assert await store.get_group_profile("10001") is None
        assert await store.get_group_profile("   ") is None
        assert await store.render_group_profile("10001") == ""

        with pytest.raises(ValueError):
            await store.patch_group_profile("   ", interaction_style="轻松")

        created = await store.patch_group_profile(
            "10001",
            interaction_style="  轻松幽默  ",
            frequent_topics="Python、游戏",
            now=1000,
        )
        assert created.group_id == "10001"
        assert created.interaction_style == "轻松幽默"
        assert created.frequent_topics == "Python、游戏"
        assert created.taboos_and_boundaries == ""
        assert created.member_notes == ""
        assert created.updated_at == 1000

        # Partial patch preserves unspecified fields; empty string clears a field.
        patched = await store.patch_group_profile(
            "10001",
            frequent_topics="",
            taboos_and_boundaries="不聊政治",
            member_notes="群主爱喝咖啡",
            now=2000,
        )
        assert patched.interaction_style == "轻松幽默"
        assert patched.frequent_topics == ""
        assert patched.taboos_and_boundaries == "不聊政治"
        assert patched.member_notes == "群主爱喝咖啡"
        assert patched.updated_at == 2000

        rendered = await store.render_group_profile("10001")
        assert rendered.startswith("## 当前群聊画像\n")
        assert "- 相处风格：轻松幽默" in rendered
        assert "常见话题" not in rendered
        assert "- 禁忌与边界：不聊政治" in rendered
        assert "- 成员备注：群主爱喝咖啡" in rendered
        assert "2000" not in rendered

        # Clearing all fields renders empty string.
        await store.patch_group_profile(
            "10001",
            interaction_style="",
            taboos_and_boundaries="",
            member_notes="",
            now=3000,
        )
        assert await store.render_group_profile("10001") == ""


async def test_user_profile_aliases_and_physical_private_notes_isolation() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(db, MemoryConfig())
        await store.init()

        assert await store.get_user_profile("u1") is None
        assert await store.get_public_user_profile("u1") is None
        assert await store.render_private_profile("u1") == ""
        assert await store.render_public_user_hint("u1") == ""

        with pytest.raises(TypeError):
            await store.patch_user_profile("u1", aliases="not-a-list")
        with pytest.raises(TypeError):
            await store.patch_user_profile("u1", add_aliases="not-a-list")

        # Profile with ONLY private_notes: DM sees it, public hint MUST be empty.
        secret_only = await store.patch_user_profile(
            "u1",
            private_notes="绝密私聊心事_SECRET_XYZ",
            now=100,
        )
        assert isinstance(secret_only, UserProfile)
        assert secret_only.private_notes == "绝密私聊心事_SECRET_XYZ"

        pub_only = await store.get_public_user_profile("u1")
        assert isinstance(pub_only, PublicUserProfile)
        assert not hasattr(pub_only, "private_notes")
        assert await store.render_public_user_hint("u1") == ""
        dm_md = await store.render_private_profile("u1")
        assert dm_md.startswith("## 当前私聊对象画像\n")
        assert "绝密私聊心事_SECRET_XYZ" in dm_md
        assert "100" not in dm_md

        # Patch public fields + alias deduplication; private_notes preserved.
        patched = await store.patch_user_profile(
            "u1",
            aliases=["  小王 ", "Alice", "alice", "", "  "],
            interaction_style="直爽",
            topic_preferences="Rust、摄影",
            relationship="老朋友",
            recent_focus="准备面试",
            now=200,
        )
        assert patched.aliases == ("小王", "Alice")
        assert patched.private_notes == "绝密私聊心事_SECRET_XYZ"

        # add_aliases appends deduplicated entries while keeping existing ones.
        added = await store.patch_user_profile(
            "u1",
            add_aliases=["ALICE", "老王", "小王"],
            now=300,
        )
        assert added.aliases == ("小王", "Alice", "老王")

        pub = await store.get_public_user_profile("u1")
        assert pub is not None
        assert not hasattr(pub, "private_notes")
        assert pub.aliases == ("小王", "Alice", "老王")
        assert pub.interaction_style == "直爽"

        hint = await store.render_public_user_hint("u1")
        assert "小王、Alice、老王" in hint
        assert "直爽" in hint
        assert "Rust、摄影" in hint
        assert "老朋友" in hint
        assert "准备面试" in hint
        assert "绝密私聊心事_SECRET_XYZ" not in hint
        assert "300" not in hint


async def test_memory_crud_constraints_reinforce_supersede_and_delete() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(db, MemoryConfig())
        await store.init()

        # Python-level validation for invariants.
        with pytest.raises(ValueError):
            await store.add_memory(
                kind="fact",
                content="bad private with group",
                visibility="private",
                owner_user_id="u1",
                source_group_id="g1",
            )
        with pytest.raises(ValueError):
            await store.add_memory(
                kind="fact",
                content="bad private without owner",
                visibility="private",
                owner_user_id="  ",
                source_group_id=None,
            )
        with pytest.raises(ValueError):
            await store.add_memory(
                kind="fact",
                content="bad group_only without group",
                visibility="group_only",
                owner_user_id="u1",
                source_group_id=None,
            )
        with pytest.raises(ValueError):
            await store.add_memory(
                kind="fact",
                content="orphan public",
                visibility="public",
                owner_user_id=None,
                source_group_id=" ",
            )

        # Raw SQLite CHECK constraints enforce the same invariants directly.
        invalid_sqls: list[tuple[str, tuple[Any, ...]]] = [
            # private with non-null source_group_id
            (
                "INSERT INTO memory_items (kind, content, owner_user_id, "
                "source_group_id, visibility, created_at, updated_at) "
                "VALUES ('fact', 'x', 'u1', 'g1', 'private', 1, 1)",
                (),
            ),
            # group_only with null source_group_id
            (
                "INSERT INTO memory_items (kind, content, owner_user_id, "
                "source_group_id, visibility, created_at, updated_at) "
                "VALUES ('fact', 'x', 'u1', NULL, 'group_only', 1, 1)",
                (),
            ),
            # both owner and group null
            (
                "INSERT INTO memory_items (kind, content, owner_user_id, "
                "source_group_id, visibility, created_at, updated_at) "
                "VALUES ('fact', 'x', NULL, NULL, 'public', 1, 1)",
                (),
            ),
            # invalid kind
            (
                "INSERT INTO memory_items (kind, content, owner_user_id, "
                "source_group_id, visibility, created_at, updated_at) "
                "VALUES ('rumor', 'x', 'u1', NULL, 'private', 1, 1)",
                (),
            ),
            # confidence > 1.0
            (
                "INSERT INTO memory_items (kind, content, owner_user_id, "
                "source_group_id, visibility, confidence, created_at, updated_at) "
                "VALUES ('fact', 'x', 'u1', NULL, 'private', 1.2, 1, 1)",
                (),
            ),
        ]
        for sql, params in invalid_sqls:
            with pytest.raises(sqlite3.IntegrityError):
                await db.execute(sql, params)

        # Add, get, reinforce, supersede chain, rollback, and delete.
        m1 = await store.add_memory(
            kind="preference",
            content="小明喜欢喝拿铁",
            visibility="public",
            owner_user_id="u1",
            source_group_id="g1",
            confidence=0.92,
            source_msg_start_id=10,
            source_msg_end_id=12,
            event_time="2026-11-01",
            now=1000,
        )
        item1 = await store.get_memory(m1)
        assert item1 is not None
        assert item1.status == "active"
        assert item1.confidence == pytest.approx(0.92)
        assert item1.evidence_count == 1
        assert item1.event_time == "2026-11-01"

        assert (
            await store.reinforce_memory(
                m1,
                confidence_boost=0.20,
                source_msg_end_id=20,
                now=1100,
            )
            is True
        )
        item1_r = await store.get_memory(m1)
        assert item1_r is not None
        assert item1_r.evidence_count == 2
        assert item1_r.confidence == pytest.approx(1.0)
        assert item1_r.source_msg_end_id == 20
        assert item1_r.updated_at == 1100

        # Self-referencing superseded_by blocked by CHECK constraint.
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(
                "UPDATE memory_items SET superseded_by = id WHERE id = ?",
                (m1,),
            )

        # Supersede m1 -> m2 -> m3.
        m2 = await store.supersede_memory(
            m1,
            kind="preference",
            content="小明改喝美式了",
            visibility="public",
            owner_user_id="u1",
            source_group_id="g1",
            confidence=0.8,
            now=1200,
        )
        old_m1 = await store.get_memory(m1)
        cur_m2 = await store.get_memory(m2)
        assert old_m1 is not None and old_m1.status == "superseded"
        assert old_m1.superseded_by == m2
        assert cur_m2 is not None and cur_m2.status == "active"

        # Reinforcing or re-superseding inactive m1 fails cleanly.
        assert await store.reinforce_memory(m1) is False
        with pytest.raises(ValueError):
            await store.supersede_memory(
                m1,
                kind="preference",
                content="duplicate supersede",
                visibility="public",
                owner_user_id="u1",
                source_group_id="g1",
            )

        # Atomic rollback check: force UPDATE failure via temporary trigger.
        await db.execute(
            """
            CREATE TRIGGER fail_supersede_update
            BEFORE UPDATE ON memory_items
            WHEN NEW.status = 'superseded'
            BEGIN
                SELECT RAISE(ABORT, 'simulated failure');
            END
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            await store.supersede_memory(
                m2,
                kind="preference",
                content="should be rolled back",
                visibility="public",
                owner_user_id="u1",
                source_group_id="g1",
                now=1300,
            )
        await db.execute("DROP TRIGGER fail_supersede_update")

        # Verify rollback: m2 still active, no leaked row.
        after_rb = await store.get_memory(m2)
        assert after_rb is not None and after_rb.status == "active"
        count_row = await db.fetchone("SELECT COUNT(*) AS cnt FROM memory_items")
        assert count_row is not None and int(count_row["cnt"]) == 2

        # Soft delete.
        assert await store.delete_memory(m2, now=1400) is True
        assert await store.delete_memory(m2, now=1500) is False
        deleted_m2 = await store.get_memory(m2)
        assert deleted_m2 is not None and deleted_m2.status == "deleted"


async def test_strict_privacy_isolation_across_groups_and_private_chats() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(db, MemoryConfig())
        await store.init()

        # Every memory contains the shared keyword "咖啡秘密" so lexical search
        # would match them all if SQL privacy filters ever leaked.
        id_u1_private = await store.add_memory(
            kind="fact",
            content="咖啡秘密：U1私聊吐槽老板",
            visibility="private",
            owner_user_id="u1",
            source_group_id=None,
            now=100,
        )
        id_u2_private = await store.add_memory(
            kind="fact",
            content="咖啡秘密：U2私聊买房计划",
            visibility="private",
            owner_user_id="u2",
            source_group_id=None,
            now=101,
        )
        id_ga_only = await store.add_memory(
            kind="event",
            content="咖啡秘密：A群内部聚餐暗号",
            visibility="group_only",
            owner_user_id="u1",
            source_group_id="gA",
            now=102,
        )
        id_ga_public_u1 = await store.add_memory(
            kind="preference",
            content="咖啡秘密：U1公开偏好手冲耶加雪菲",
            visibility="public",
            owner_user_id="u1",
            source_group_id="gA",
            now=103,
        )
        id_ga_public_group = await store.add_memory(
            kind="fact",
            content="咖啡秘密：A群每周五下午一起点咖啡",
            visibility="public",
            owner_user_id=None,
            source_group_id="gA",
            now=104,
        )
        id_gb_only = await store.add_memory(
            kind="fact",
            content="咖啡秘密：B群专属项目代号",
            visibility="group_only",
            owner_user_id="u2",
            source_group_id="gB",
            now=105,
        )

        # 1) Group A (even with U1 active): sees gA group_only + gA public;
        # NEVER sees U1/U2 private or gB group_only.
        ga_list = await store.list_active_memories(
            group_id="gA",
            active_speaker_ids=["u1"],
        )
        assert {m.id for m in ga_list} == {
            id_ga_only,
            id_ga_public_u1,
            id_ga_public_group,
        }
        ga_search = await store.search_memories(
            "咖啡秘密",
            group_id="gA",
            active_speaker_ids=["u1"],
            limit=20,
        )
        assert {h.item.id for h in ga_search} == {
            id_ga_only,
            id_ga_public_u1,
            id_ga_public_group,
        }

        # 2) Group B with U1 active speaker: sees gB group_only + U1's public
        # memory from gA; NEVER sees U1 private or gA group_only or gA group-level.
        gb_u1_list = await store.list_active_memories(
            group_id="gB",
            active_speaker_ids=["u1", " ", ""],
        )
        assert {m.id for m in gb_u1_list} == {id_gb_only, id_ga_public_u1}
        gb_u1_search = await store.search_memories(
            "咖啡秘密",
            group_id="gB",
            active_speaker_ids=["u1"],
            limit=20,
        )
        assert {h.item.id for h in gb_u1_search} == {id_gb_only, id_ga_public_u1}

        # 3) Group B with NO non-empty active_speaker_ids: IN () never emitted;
        # sees only gB items.
        gb_empty_list = await store.list_active_memories(
            group_id="gB",
            active_speaker_ids=["", "   "],
        )
        assert {m.id for m in gb_empty_list} == {id_gb_only}

        # 4) U1 DM: sees U1 private + U1 public; NEVER sees gA group_only or U2 private.
        u1_dm_list = await store.list_active_memories(group_id=None, user_id="u1")
        assert {m.id for m in u1_dm_list} == {id_u1_private, id_ga_public_u1}
        u1_dm_search = await store.search_memories(
            "咖啡秘密",
            group_id=None,
            user_id="u1",
            limit=20,
        )
        assert {h.item.id for h in u1_dm_search} == {id_u1_private, id_ga_public_u1}

        # 5) U2 DM: sees only U2 private.
        u2_dm_search = await store.search_memories(
            "咖啡秘密",
            group_id=None,
            user_id="u2",
            limit=20,
        )
        assert {h.item.id for h in u2_dm_search} == {id_u2_private}

        # Invalid scopes raise ValueError.
        with pytest.raises(ValueError):
            await store.list_active_memories(group_id=None, user_id=None)
        with pytest.raises(ValueError):
            await store.search_memories("咖啡秘密", group_id="   ", user_id="u1")


async def test_search_memories_3stage_vector_jev_rerank_and_no_tx_during_http() -> None:
    embed_vectors = {
        "猫": [1.0, 0.0, 0.0],
        "小明养了一只布偶猫": [0.98, 0.2, 0.0],
        "小明的猫叫年糕": [0.85, 0.5, 0.0],
        "完全无关的天气记录": [0.0, 0.0, 1.0],
        "关于猫咪玩具的讨论": [0.95, 0.3, 0.0],
    }

    async with Database(":memory:") as db:
        in_tx_observations: list[bool] = []

        def embed_handler(request: httpx.Request) -> httpx.Response:
            in_tx_observations.append(id(db) in _ACTIVE_TX_DBS.get())
            body = json.loads(request.content.decode("utf-8"))
            inputs: list[str] = body["input"]
            data = [
                {"index": i, "embedding": embed_vectors[text]}
                for i, text in enumerate(inputs)
            ]
            return httpx.Response(200, json={"data": data})

        def jev_handler(request: httpx.Request) -> httpx.Response:
            in_tx_observations.append(id(db) in _ACTIVE_TX_DBS.get())
            body = json.loads(request.content.decode("utf-8"))
            cands: list[str] = body["state"]["candidates"]
            answers: dict[str, Any] = {}
            for idx, text in enumerate(cands):
                if "年糕" in text:
                    answers[f"m{idx}"] = {"noul": 0.95}
                elif "布偶猫" in text:
                    answers[f"m{idx}"] = {"noul": 0.72}
                else:
                    answers[f"m{idx}"] = {"noul": 0.20}
            return httpx.Response(
                200,
                json={"model": "typesafe/jev-1.14", "answers": answers},
            )

        emb_cfg = EmbeddingConfig(
            base_url="https://embed.example/v1",
            api_key="sk-emb",
            model="test-embed",
        )
        jev_cfg = JevConfig(
            base_url="https://openrouter.ai/api/alpha/decisions",
            api_key="sk-jev",
            model="typesafe/jev-1.14",
        )
        mem_cfg = MemoryConfig(
            vector_top_k=10,
            min_vector_similarity=0.30,
            rerank_threshold=0.50,
            max_recall_items=5,
        )

        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(embed_handler)) as emb_http,
            httpx.AsyncClient(transport=httpx.MockTransport(jev_handler)) as jev_http,
        ):
            emb_client = EmbeddingClient(emb_cfg, http_client=emb_http)
            jev_client = JevClient(jev_cfg, http_client=jev_http)
            store = MemoryStore(
                db,
                mem_cfg,
                embedding_client=emb_client,
                jev_client=jev_client,
            )
            await store.init()

            id_ragdoll = await store.add_memory(
                kind="fact",
                content="小明养了一只布偶猫",
                visibility="group_only",
                source_group_id="g1",
                now=100,
            )
            id_niangao = await store.add_memory(
                kind="fact",
                content="小明的猫叫年糕",
                visibility="group_only",
                source_group_id="g1",
                now=101,
            )
            id_weather = await store.add_memory(
                kind="event",
                content="完全无关的天气记录",
                visibility="group_only",
                source_group_id="g1",
                now=102,
            )
            id_toy = await store.add_memory(
                kind="fact",
                content="关于猫咪玩具的讨论",
                visibility="group_only",
                source_group_id="g1",
                now=103,
            )

            hits = await store.search_memories("猫", group_id="g1", now=9999)

            # Verify no HTTP call ever ran inside a SQLite transaction.
            assert in_tx_observations
            assert all(in_tx is False for in_tx in in_tx_observations)

            # Jev rerank puts 年糕 (0.95) above 布偶猫 (0.72), drops toy (0.20)
            # and weather was already filtered out by vector similarity < 0.30.
            assert [h.item.id for h in hits] == [id_niangao, id_ragdoll]
            assert hits[0].rerank_score == pytest.approx(0.95)
            assert hits[1].rerank_score == pytest.approx(0.72)
            assert hits[0].vector_score is not None and hits[0].vector_score > 0.8
            assert hits[0].item.last_accessed_at == 9999
            assert hits[1].item.last_accessed_at == 9999

            # Confirm SQLite last_accessed_at updated ONLY for returned rows.
            assert (await store.get_memory(id_niangao)).last_accessed_at == 9999  # type: ignore[union-attr]
            assert (await store.get_memory(id_ragdoll)).last_accessed_at == 9999  # type: ignore[union-attr]
            assert (await store.get_memory(id_weather)).last_accessed_at == 0  # type: ignore[union-attr]
            assert (await store.get_memory(id_toy)).last_accessed_at == 0  # type: ignore[union-attr]


async def test_search_memories_jev_openai_decision_degradation_and_fallbacks() -> None:
    async with Database(":memory:") as db:
        # 1) OpenAI Decisions rerank + unembedded & model-mismatch lexical fallback.
        def emb_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            )

        jev_responses = iter(
            [
                # First search: OpenAI decision protocol succeeds
                httpx.Response(
                    200,
                    json={
                        "model": "gpt-4.1-mini",
                        "answers": [
                            {"type": "predicate", "name": "m0", "probability": 0.65},
                            {"type": "predicate", "name": "m1", "probability": 0.88},
                            {"type": "predicate", "name": "m2", "probability": 0.77},
                        ],
                    },
                ),
                # Second search: Jev 500 -> degrades gracefully to Stage 1 order
                httpx.Response(500, text="boom"),
            ]
        )

        emb_cfg = EmbeddingConfig(
            base_url="https://embed.example/v1",
            api_key="sk-e",
            model="new-embed",
        )
        jev_cfg = JevConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-oa",
            model="gpt-4.1-mini",
        )
        mem_cfg = MemoryConfig(
            min_vector_similarity=0.25,
            rerank_threshold=0.50,
            max_recall_items=5,
        )

        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(emb_handler)) as emb_http,
            httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: next(jev_responses))
            ) as jev_http,
        ):
            store = MemoryStore(
                db,
                mem_cfg,
                embedding_client=EmbeddingClient(emb_cfg, http_client=emb_http),
                jev_client=JevClient(jev_cfg, http_client=jev_http),
            )
            await store.init()

            # Row with compatible embedding
            id_vec = await store.add_memory(
                kind="fact",
                content="他用 Python 写机器人",
                visibility="private",
                owner_user_id="u1",
                embedding=_make_embedding([0.9, 0.1], model="new-embed"),
                now=10,
            )
            # Row with NO embedding (lexical fallback should rescue it!)
            id_no_emb = await store.add_memory(
                kind="fact",
                content="他最喜欢的语言是 python",
                visibility="private",
                owner_user_id="u1",
                embedding=None,
                now=11,
            )
            # Manually strip embedding so it simulates a previously unembedded row
            await db.execute(
                "UPDATE memory_items SET embedding = NULL, embedding_model = NULL "
                "WHERE id = ?",
                (id_no_emb,),
            )
            # Row with OLD/mismatched embedding model tag (lexical fallback rescues!)
            id_old_model = await store.add_memory(
                kind="fact",
                content="他昨天在学 Python 异步编程",
                visibility="private",
                owner_user_id="u1",
                embedding=_make_embedding([1.0, 0.0, 0.0], model="old-embed"),
                now=12,
            )

            hits_ok = await store.search_memories(
                "Python",
                group_id=None,
                user_id="u1",
                now=500,
            )
            # Stage 1 order: [id_vec (vector), id_no_emb/id_old_model (lexical)].
            # Jev OpenAI decision gave m1=0.88, m2=0.77, m0=0.65.
            assert len(hits_ok) == 3
            assert hits_ok[0].rerank_score == pytest.approx(0.88)
            assert hits_ok[2].item.id == id_vec
            assert hits_ok[2].vector_score is not None
            assert {h.item.id for h in hits_ok} == {id_vec, id_no_emb, id_old_model}

            # Second search: Jev degrades -> keeps Stage 1 order with vector row first!
            hits_deg = await store.search_memories(
                "Python",
                group_id=None,
                user_id="u1",
                limit=2,
                now=600,
            )
            assert len(hits_deg) == 2
            assert hits_deg[0].item.id == id_vec
            assert hits_deg[0].vector_score is not None
            assert all(h.rerank_score is None for h in hits_deg)

        # Disabled config or blank query returns [] immediately.
        disabled_store = MemoryStore(db, MemoryConfig(enabled=False))
        assert (
            await disabled_store.search_memories("Python", group_id=None, user_id="u1")
            == []
        )
        lexical_store = MemoryStore(db, MemoryConfig())
        assert (
            await lexical_store.search_memories("   ", group_id=None, user_id="u1")
            == []
        )
        assert (
            await lexical_store.search_memories(
                "Python", group_id=None, user_id="u1", limit=0
            )
            == []
        )


async def test_group_jargon_upsert_case_insensitive_and_boundary_matching() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(
            db,
            MemoryConfig(jargon_min_confidence=0.60, jargon_max_matches=3),
        )
        await store.init()

        j_go_1 = await store.upsert_jargon(
            "g1",
            "go",
            "一起开黑打游戏",
            confidence=0.70,
            now=100,
        )
        # Case-insensitive upsert in same group merges & boosts confidence.
        j_go_2 = await store.upsert_jargon(
            "g1",
            "GO",
            "一起开黑打瓦罗兰特",
            confidence=0.70,
            confidence_boost=0.10,
            now=200,
        )
        assert j_go_2 == j_go_1

        # Same term in a different group is isolated.
        j_go_g2 = await store.upsert_jargon(
            "g2",
            "go",
            "Golang 编程语言",
            confidence=0.80,
            now=210,
        )
        assert j_go_g2 != j_go_1

        await store.upsert_jargon("g1", "C", "C 语言", confidence=0.85, now=220)
        await store.upsert_jargon("g1", "C++", "C++ 语言", confidence=0.82, now=230)
        await store.upsert_jargon("g1", "开黑", "组队打游戏", confidence=0.75, now=240)
        await store.upsert_jargon(
            "g1", "低置信梗", "不应触发", confidence=0.40, now=250
        )

        listed = await store.list_group_jargon("g1", min_confidence=0.60)
        assert [e.term for e in listed] == ["C", "C++", "go", "开黑"]
        go_entry = next(e for e in listed if e.id == j_go_1)
        assert go_entry.evidence_count == 2
        assert go_entry.confidence == pytest.approx(0.80)
        assert go_entry.meaning == "一起开黑打瓦罗兰特"

        # "今晚go吗" matches "go" across CJK neighbors!
        m_cjk_go = await store.match_jargon_in_text("g1", "今晚go吗？顺便开黑")
        assert [e.term for e in m_cjk_go] == ["go", "开黑"]

        # Case-insensitive ASCII match.
        m_upper_go = await store.match_jargon_in_text("g1", "今晚GO不GO")
        assert [e.term for e in m_upper_go] == ["go"]

        # "good" and "cargo" must NOT false-match "go".
        assert (
            await store.match_jargon_in_text("g1", "good morning cargo go_fast") == []
        )

        # "C++" must NOT false-match "C", while it DOES match "C++".
        m_cpp = await store.match_jargon_in_text("g1", "我在写C++代码")
        assert [e.term for e in m_cpp] == ["C++"]
        assert await store.match_jargon_in_text("g1", "我在学C#开发") == []
        m_c_only = await store.match_jargon_in_text("g1", "我在学c语言")
        assert [e.term for e in m_c_only] == ["C"]

        # Low-confidence term never matches; group isolation holds.
        assert await store.match_jargon_in_text("g1", "低置信梗") == []
        m_g2 = await store.match_jargon_in_text("g2", "今晚go吗")
        assert len(m_g2) == 1 and m_g2[0].meaning == "Golang 编程语言"

        # Respects jargon_max_matches=3 in (confidence DESC, evidence DESC, id ASC).
        m_all = await store.match_jargon_in_text("g1", "学c也学c++，今晚go开黑")
        assert [e.term for e in m_all] == ["C", "C++", "go"]


async def test_lexical_fallback_ranking_and_corrupted_embedding_handling() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(db, MemoryConfig(max_recall_items=5))
        await store.init()

        id_low = await store.add_memory(
            kind="fact",
            content="小明周末去公园散步",
            visibility="group_only",
            source_group_id="g1",
            confidence=0.60,
            now=100,
        )
        id_high = await store.add_memory(
            kind="preference",
            content="小明最喜欢在周末去吃拉面和逛公园",
            visibility="group_only",
            source_group_id="g1",
            confidence=0.95,
            now=101,
        )
        await store.add_memory(
            kind="fact",
            content="完全无关的数据库迁移日志",
            visibility="group_only",
            source_group_id="g1",
            confidence=0.99,
            now=102,
        )

        # Pure lexical search (embedding_client=None) ranks higher overlap + confidence.
        lex_hits = await store.search_memories(
            "小明周末拉面",
            group_id="g1",
            now=500,
        )
        assert [h.item.id for h in lex_hits] == [id_high, id_low]
        assert all(h.vector_score is None and h.rerank_score is None for h in lex_hits)

        # Corrupted NaN float32 blob on a matching model_tag degrades to lexical safely.
        nan_blob = b"\x00\x00\xc0\x7f\x00\x00\x80\x3f"  # [NaN, 1.0]
        await db.execute(
            "UPDATE memory_items SET embedding = ?, embedding_model = ? WHERE id = ?",
            (nan_blob, "m:2", id_high),
        )

        def emb_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            )

        emb_cfg = EmbeddingConfig(
            base_url="https://embed.example/v1",
            api_key="sk-e",
            model="m",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(emb_handler)
        ) as emb_http:
            vec_store = MemoryStore(
                db,
                MemoryConfig(),
                embedding_client=EmbeddingClient(emb_cfg, http_client=emb_http),
            )
            hits = await vec_store.search_memories("小明周末拉面", group_id="g1")
            assert [h.item.id for h in hits] == [id_high, id_low]
            assert all(h.vector_score is None for h in hits)
