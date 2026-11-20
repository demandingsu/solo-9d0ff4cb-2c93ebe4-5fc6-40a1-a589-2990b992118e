"""端到端 Web 流程测试：上传 → 勾选/剔除 → 预览 → 下载 / 冲突阻止，
以及多标签页并发下的交付一致性（旧页确认拒绝、下载中换库不中断）。"""
import asyncio
import os
import re
import sqlite3
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "data"))
    main.SESSIONS.clear()
    main.SOURCES.clear()
    return TestClient(main.app)


def _upload(client, sample_db, name="lab.db"):
    with open(sample_db, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": (name, f, "application/x-sqlite3")},
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/select"


def _preview_token(client):
    """打开预览页并取出下载表单中的状态令牌。"""
    resp = client.get("/preview")
    assert resp.status_code == 200
    m = re.search(r'name="token" value="([0-9a-f]+)"', resp.text)
    assert m, "预览页下载表单应携带状态令牌"
    return m.group(1)


def test_full_flow_download(client, sample_db, tmp_path):
    _upload(client, sample_db)

    resp = client.get("/select")
    assert resp.status_code == 200
    assert "墙体试件A" in resp.text and "背景噪声对照测量" in resp.text

    resp = client.post("/select", data={"test": ["1"]}, follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get("/preview")
    assert resp.status_code == 200
    assert "确认并下载" in resp.text
    assert "901" in resp.text  # 跨组带入的对照读数出现在预览中

    resp = client.post("/download", data={"token": _preview_token(client)})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-sqlite3")
    out = tmp_path / "downloaded.db"
    out.write_bytes(resp.content)
    conn = sqlite3.connect(out)
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    reading_ids = {r[0] for r in conn.execute("SELECT id FROM readings")}
    assert reading_ids == {101, 102, 103, 901}
    conn.close()


def test_conflict_blocks_download(client, sample_db):
    _upload(client, sample_db)
    # 勾选 T1 与 T9，并剔除被 T1 读数引用的对照读数 R901
    resp = client.post(
        "/select",
        data={"test": ["1", "9"], "excl": ["901"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    resp = client.get("/preview")
    assert resp.status_code == 200
    assert "已阻止下载" in resp.text
    assert "剔除" in resp.text  # 引用链与冲突原因展示在预览页

    resp = client.post("/download", data={"token": _preview_token(client)})
    assert resp.status_code == 409
    assert "阻止" in resp.text


def test_exclusion_of_unselected_test_reading_ignored(client, sample_db):
    """剔除未勾选试验的读数被静默忽略，不影响交付。"""
    _upload(client, sample_db)
    client.post("/select", data={"test": ["1"], "excl": ["901"]})
    resp = client.get("/preview")
    assert "确认并下载" in resp.text
    resp = client.post("/download", data={"token": _preview_token(client)})
    assert resp.status_code == 200


def test_upload_rejects_bad_file(client, tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is not sqlite")
    with open(bad, "rb") as f:
        resp = client.post("/upload", files={"file": ("bad.db", f)})
    assert resp.status_code == 400
    assert "SQLite" in resp.text


def test_upload_rejects_wrong_schema(client, tmp_path):
    wrong = tmp_path / "wrong.db"
    conn = sqlite3.connect(wrong)
    conn.execute("CREATE TABLE whatever (id INTEGER)")
    conn.commit()
    conn.close()
    with open(wrong, "rb") as f:
        resp = client.post("/upload", files={"file": ("wrong.db", f)})
    assert resp.status_code == 400
    assert "缺少数据表" in resp.text


def test_pages_require_session(client):
    assert client.get("/select", follow_redirects=False).status_code == 303
    assert client.get("/preview", follow_redirects=False).status_code == 303


def test_download_without_token_rejected(client, sample_db):
    """未携带状态令牌的确认请求一律拒绝。"""
    _upload(client, sample_db)
    client.post("/select", data={"test": ["1"]})
    client.get("/preview")
    resp = client.post("/download")
    assert resp.status_code == 409
    assert "重新预览" in resp.text


def test_stale_confirm_rejected_after_reselect(client, sample_db, tmp_path):
    """改选试验后，旧预览页的确认被拒绝并要求重新预览。"""
    _upload(client, sample_db)
    client.post("/select", data={"test": ["1"]})
    stale_token = _preview_token(client)

    # 另一标签页改选试验（T1 → T3）
    client.post("/select", data={"test": ["3"]})

    resp = client.post("/download", data={"token": stale_token})
    assert resp.status_code == 409
    assert "重新预览" in resp.text

    # 重新预览后确认成功，且交付内容按最新预览（T3 及其引用链）
    resp = client.post("/download", data={"token": _preview_token(client)})
    assert resp.status_code == 200
    out = tmp_path / "dl.db"
    out.write_bytes(resp.content)
    conn = sqlite3.connect(out)
    ids = {r[0] for r in conn.execute("SELECT id FROM readings")}
    conn.close()
    assert ids == {201, 202, 901, 902}


def test_stale_confirm_rejected_after_exclusion_change(client, sample_db):
    """调整剔除读数后，旧预览页的确认被拒绝。"""
    _upload(client, sample_db)
    client.post("/select", data={"test": ["2"]})
    stale_token = _preview_token(client)
    client.post("/select", data={"test": ["2"], "excl": ["112"]})
    resp = client.post("/download", data={"token": stale_token})
    assert resp.status_code == 409
    assert "重新预览" in resp.text


def test_stale_confirm_rejected_after_reupload(client, sample_db):
    """换库（即使文件内容相同）后，旧预览页的确认被拒绝。"""
    _upload(client, sample_db)
    client.post("/select", data={"test": ["1"]})
    stale_token = _preview_token(client)

    _upload(client, sample_db, name="new.db")  # 另一标签页重新上传

    resp = client.post("/download", data={"token": stale_token})
    assert resp.status_code == 409
    assert "重新预览" in resp.text


def test_download_cleans_up_temp_subset(client, sample_db, tmp_path):
    """传输结束后，本次生成的临时子库目录被清理。"""
    _upload(client, sample_db)
    client.post("/select", data={"test": ["1"]})
    resp = client.post("/download", data={"token": _preview_token(client)})
    assert resp.status_code == 200
    jobs_dir = tmp_path / "data" / "jobs"
    assert not list(jobs_dir.iterdir())  # 作业目录已清空


def test_source_dir_removed_only_after_lease(tmp_path, monkeypatch):
    """换下的源库在下载租约归零前保留，归零后删除。"""
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "data"))
    main.SOURCES.clear()
    src_dir = main._source_dir("s1")
    os.makedirs(src_dir)
    main.SOURCES["s1"] = main._SourceEntry("a.db")

    assert main._acquire_lease("s1")
    main._release_source("s1")  # 换库：摘除会话引用
    assert os.path.isdir(src_dir)  # 租约未结束，源库保留
    main._release_lease("s1")
    assert not os.path.exists(src_dir)  # 租约归零，源库删除
    assert "s1" not in main.SOURCES


def test_download_in_progress_survives_reupload(tmp_path, monkeypatch, sample_db):
    """确认受理后，另一页面换库不打断本次传输；子库按该页预览生成。

    传输结束后再验证：临时子库已清理，换下且不再使用的源库已删除。
    """
    data_dir = tmp_path / "data"
    monkeypatch.setattr(main, "DATA_DIR", str(data_dir))
    main.SESSIONS.clear()
    main.SOURCES.clear()

    build_started = threading.Event()
    finish_build = threading.Event()
    real_build = main.build_subset

    def slow_build(src, out, res):
        # 子库生成开始（源库租约已持有）后暂停，等换库完成再继续
        build_started.set()
        assert finish_build.wait(10)
        return real_build(src, out, res)

    monkeypatch.setattr(main, "build_subset", slow_build)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            with open(sample_db, "rb") as f:
                r = await c.post("/upload", files={"file": ("lab.db", f)})
            assert r.status_code == 303
            await c.post("/select", data={"test": ["1"]})
            r = await c.get("/preview")
            token = re.search(r'name="token" value="([0-9a-f]+)"', r.text).group(1)

            # 确认下载（受理），子库生成期间另一标签页换库
            dl = asyncio.create_task(c.post("/download", data={"token": token}))
            while not build_started.is_set():
                await asyncio.sleep(0.01)
            with open(sample_db, "rb") as f:
                r = await c.post("/upload", files={"file": ("new.db", f)})
            assert r.status_code == 303  # 换库成功，未被下载阻塞
            finish_build.set()
            resp = await dl
            assert resp.status_code == 200
            return resp.content

    content = asyncio.run(scenario())

    # 交付内容按该页预览生成（T1 及其跨组引用 R901），未受换库影响
    out = tmp_path / "dl.db"
    out.write_bytes(content)
    conn = sqlite3.connect(out)
    ids = {r[0] for r in conn.execute("SELECT id FROM readings")}
    conn.close()
    assert ids == {101, 102, 103, 901}

    # 传输结束：临时子库已清理；换下且不再使用的源库已删除，只剩新库
    assert not list((data_dir / "jobs").iterdir())
    sources = list((data_dir / "sources").iterdir())
    assert len(sources) == 1
