"""隔声试验数据交付复核 Web 应用（FastAPI）。

流程：上传 SQLite 库 → 勾选试验 / 剔除异常读数 → 预览引用解析结果
（含跨组拉入链与冲突）→ 确认后下载 SQLite 子库。

交付一致性约定（多标签页/并发场景）：
- 预览页携带状态令牌（源库身份 + 勾选试验 + 剔除读数的内容摘要）；
  确认下载时先校验令牌，若另一页面已改选、调整剔除或换库，
  则拒绝发文件并提示重新预览；
- 确认受理后按该页预览时的状态快照生成子库，生成期间源库被租约保护，
  另一页面可同时改选或换库，不影响本次传输；
- 传输结束或失败后清理本次生成的临时子库；被换下且不再被引用的
  源库在租约归零后删除。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .core import build_subset, load_model, resolve, validate_schema

DATA_DIR = os.environ.get("DATA_DIR", "./data")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))


def _source_dir(source_id: str) -> str:
    return os.path.join(DATA_DIR, "sources", source_id)


def _source_path(source_id: str) -> str:
    return os.path.join(_source_dir(source_id), "source.db")


def _jobs_dir() -> str:
    return os.path.join(DATA_DIR, "jobs")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # 上次运行可能遗留未传完的临时子库，启动时一并清理；
    # 源库保留（会话在内存中，重启后失效，源库文件由运维清理）。
    shutil.rmtree(_jobs_dir(), ignore_errors=True)
    yield


app = FastAPI(title="隔声试验数据交付复核", lifespan=_lifespan)
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
)


@dataclass
class SessionState:
    source_id: str
    source_name: str
    selected_tests: set[int] = field(default_factory=set)
    excluded_readings: set[int] = field(default_factory=set)

    @property
    def db_path(self) -> str:
        return _source_path(self.source_id)


@dataclass
class _SourceEntry:
    """一个已上传源库的生命周期：会话引用 + 进行中下载租约。"""

    name: str
    session_attached: bool = True  # 是否有会话正引用该源库
    leases: int = 0  # 进行中的子库生成数（租约）


# 单进程内存会话；重启即失效（上传文件保留在 DATA_DIR 下，由运维清理）
SESSIONS: dict[str, SessionState] = {}
# source_id -> _SourceEntry；会话引用与下载租约共同决定源库目录何时可删除
SOURCES: dict[str, _SourceEntry] = {}
SOURCES_LOCK = threading.Lock()


def _drop_source_dir(source_id: str) -> None:
    shutil.rmtree(_source_dir(source_id), ignore_errors=True)


def _release_source(source_id: str) -> None:
    """会话换库时摘除会话引用；若无进行中租约则删除源目录。"""
    with SOURCES_LOCK:
        ent = SOURCES.get(source_id)
        if ent is None:
            return
        ent.session_attached = False
        removable = ent.leases == 0
        if removable:
            SOURCES.pop(source_id, None)
    if removable:
        _drop_source_dir(source_id)


def _acquire_lease(source_id: str) -> bool:
    """为子库生成租用源库，阻止换库清理将其删除；源库已退役则失败。"""
    with SOURCES_LOCK:
        ent = SOURCES.get(source_id)
        if ent is None or not ent.session_attached:
            return False
        ent.leases += 1
        return True


def _release_lease(source_id: str) -> None:
    """归还租约；源库已换下且无其他租约时删除源目录。"""
    with SOURCES_LOCK:
        ent = SOURCES.get(source_id)
        if ent is None:
            return
        ent.leases -= 1
        removable = ent.leases == 0 and not ent.session_attached
        if removable:
            SOURCES.pop(source_id, None)
    if removable:
        _drop_source_dir(source_id)


def _state_token(source_id: str, selected: set[int], excluded: set[int]) -> str:
    """预览状态令牌：源库身份 + 勾选试验 + 剔除读数的内容摘要。

    任一变更（改选、调整剔除、换库）都会改变令牌，使旧预览页的
    确认请求被拒绝；内容相同则令牌相同，重复确认不受影响。
    """
    h = hashlib.sha256()
    h.update(b"src:" + source_id.encode())
    h.update(b"|tests:" + ",".join(map(str, sorted(selected))).encode())
    h.update(b"|excl:" + ",".join(map(str, sorted(excluded))).encode())
    return h.hexdigest()


def _get_session(request: Request) -> SessionState | None:
    sid = request.cookies.get("sid")
    return SESSIONS.get(sid) if sid else None


def _load(st: SessionState):
    conn = sqlite3.connect(f"file:{st.db_path}?mode=ro", uri=True)
    try:
        return load_model(conn)
    finally:
        conn.close()


class _JobFileResponse(FileResponse):
    """发送本次生成的临时子库；传输结束（含客户端断连、发送失败）后清理作业目录。"""

    def __init__(self, path: str, job_dir: str, **kwargs):
        super().__init__(path, **kwargs)
        self._job_dir = job_dir

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            shutil.rmtree(self._job_dir, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "upload.html", {"error": None}
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile):
    error: str | None = None
    if not file.filename:
        error = "未选择文件"
    data = await file.read()
    if error is None and len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        error = f"文件超过大小限制（{MAX_UPLOAD_MB} MB）"

    if error is None:
        # 先落盘并校验新库，通过后再换下旧库，避免无效上传弄丢现有会话
        source_id = uuid.uuid4().hex
        os.makedirs(_source_dir(source_id), exist_ok=True)
        path = _source_path(source_id)
        with open(path, "wb") as f:
            f.write(data)
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            errors = validate_schema(conn)
            conn.close()
        except sqlite3.DatabaseError:
            errors = ["文件不是有效的 SQLite 数据库"]
        if errors:
            shutil.rmtree(_source_dir(source_id), ignore_errors=True)
            return templates.TemplateResponse(
                request,
                "upload.html",
                {"error": "；".join(errors)},
                status_code=400,
            )

        # 换下旧库：仅摘除会话引用；若旧库正被下载租约使用，
        # 实际删除推迟到租约归零，不打断已受理的传输
        old = _get_session(request)
        if old:
            SESSIONS.pop(request.cookies.get("sid"), None)
            _release_source(old.source_id)
        with SOURCES_LOCK:
            SOURCES[source_id] = _SourceEntry(file.filename or "source.db")

        sid = uuid.uuid4().hex
        SESSIONS[sid] = SessionState(source_id, file.filename or "source.db")
        resp = RedirectResponse("/select", status_code=303)
        resp.set_cookie("sid", sid, httponly=True)
        return resp

    return templates.TemplateResponse(
        request, "upload.html", {"error": error}, status_code=400
    )


# 读数表展示时的优先列序，其余列按出现顺序排后
_PREFERRED_READING_COLS = [
    "id", "kind", "frequency_hz", "level_db",
    "control_reading_id", "calibration_id", "note",
]


@app.get("/select", response_class=HTMLResponse)
def select_page(request: Request):
    st = _get_session(request)
    if not st:
        return RedirectResponse("/", status_code=303)
    model = _load(st)

    headers: list[str] = []
    for r in model.readings.values():
        for col in r:
            if col not in headers:
                headers.append(col)
    headers = [c for c in _PREFERRED_READING_COLS if c in headers] + [
        c for c in headers if c not in _PREFERRED_READING_COLS
    ]

    groups = []
    for gid in sorted(model.groups):
        g = model.groups[gid]
        tests = []
        for t in sorted(model.tests.values(), key=lambda x: x["id"]):
            if t["group_id"] != gid:
                continue
            readings = [
                {
                    "id": r["id"],
                    "cells": [r.get(c) for c in headers],
                    "excluded": r["id"] in st.excluded_readings,
                }
                for r in (
                    model.readings[rid]
                    for rid in model.readings_by_test.get(t["id"], [])
                )
            ]
            tests.append(
                {"row": t, "readings": readings, "selected": t["id"] in st.selected_tests}
            )
        groups.append({"row": g, "tests": tests})

    return templates.TemplateResponse(
        request,
        "select.html",
        {
            "groups": groups,
            "headers": headers,
            "source": st.source_name,
        },
    )


@app.post("/select")
async def select_save(request: Request):
    st = _get_session(request)
    if not st:
        return RedirectResponse("/", status_code=303)
    form = await request.form()
    model = _load(st)

    selected = {int(v) for v in form.getlist("test")} & set(model.tests)
    # 剔除仅对所选试验内的读数生效，其余静默忽略
    excluded = {
        int(v)
        for v in form.getlist("excl")
        if (r := model.readings.get(int(v))) and r["test_id"] in selected
    }
    st.selected_tests = selected
    st.excluded_readings = excluded
    return RedirectResponse("/preview", status_code=303)


@app.get("/preview", response_class=HTMLResponse)
def preview(request: Request):
    st = _get_session(request)
    if not st:
        return RedirectResponse("/", status_code=303)
    model = _load(st)
    res = resolve(model, st.selected_tests, st.excluded_readings)
    return templates.TemplateResponse(
        request,
        "preview.html",
        {
            "res": res,
            "model": model,
            "selected": sorted(st.selected_tests),
            "excluded": sorted(st.excluded_readings),
            "pulled_in": sorted(res.pulled_in.items()),
            "source": st.source_name,
            "token": _state_token(st.source_id, st.selected_tests, st.excluded_readings),
        },
    )


_STALE_HINT = (
    "<h2>请重新预览后再下载</h2>"
    "<p>该预览页已过期：试验勾选、剔除读数或源库已在其他页面变更，"
    "为避免交付与预览不符的子库，本次下载已取消、未发送文件。</p>"
    "<p>请返回 <a href='/preview'>交付预览</a> 核对最新内容后重新确认，"
    "或回 <a href='/select'>选择页</a> 修改。</p>"
)


@app.post("/download")
def download(request: Request, token: str = Form("")):
    st = _get_session(request)
    if not st:
        return RedirectResponse("/", status_code=303)

    # 快照当前状态：令牌校验与子库生成都以快照为准。受理后其他页面
    # 改选、调整剔除或换库，均不影响本次交付内容。
    source_id = st.source_id
    selected = set(st.selected_tests)
    excluded = set(st.excluded_readings)
    if not hmac.compare_digest(token, _state_token(source_id, selected, excluded)):
        return HTMLResponse(_STALE_HINT, status_code=409)

    # 租用源库直至子库生成完毕，期间换库不会删除源文件
    if not _acquire_lease(source_id):
        return HTMLResponse(_STALE_HINT, status_code=409)
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(_jobs_dir(), job_id)
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, "subset.db")
    try:
        try:
            model = _load(st)
            res = resolve(model, selected, excluded)
            if res.ok:
                build_subset(st.db_path, out_path, res)
        finally:
            _release_lease(source_id)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    if not res.ok:
        shutil.rmtree(job_dir, ignore_errors=True)
        return HTMLResponse(
            "<h2>已阻止下载</h2><p>存在未解决的引用冲突（悬空或指向已剔除读数）。"
            "请返回 <a href='/preview'>预览</a> 查看引用链，"
            "或回 <a href='/select'>选择页</a> 修改。</p>",
            status_code=409,
        )
    name = os.path.splitext(st.source_name)[0] + "_subset.db"
    return _JobFileResponse(
        out_path, job_dir, filename=name, media_type="application/x-sqlite3"
    )
