# app.py — ManeKrit (sharded-chunks loader, mobile login, admin, SQLite history)
# Safe dotenv import so Streamlit Cloud won't crash if python-dotenv isn't present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os
import re
import io
import json
import gzip
import time
import sqlite3
import requests
import numpy as np
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Set, Tuple

# OpenAI client
try:
    from openai import OpenAI
except Exception:
    # show friendly message in UI later if missing — but import failure here should stop
    raise

# Chroma (in-memory)
import chromadb
from chromadb.config import Settings

# -------------------------------
# 0) Config / Secrets
# -------------------------------
def get_config(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name)  # type: ignore[attr-defined]
        if v:
            return str(v)
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY   = get_config("OPENAI_API_KEY")
GDRIVE_API_KEY   = get_config("GDRIVE_API_KEY")
GDRIVE_FOLDER_ID = get_config("GDRIVE_FOLDER_ID")
ADMIN_PASSWORD   = get_config("ADMIN_PASSWORD", "adminpass")

CHAT_MODEL  = "gpt-5"
EMBED_MODEL = "text-embedding-3-small"
TOP_K       = 5
TZ_OFFSET   = 5.5  # IST offset for charts

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# 1) Persistence (SQLite)
# -------------------------------
DB_PATH = "conversations.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mobile TEXT NOT NULL,
            ts INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_conv_mobile_ts ON conversations (mobile, ts)")
    conn.commit(); conn.close()

def log_event(mobile: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO conversations (mobile, ts, role, content) VALUES (?, ?, ?, ?)",
              (mobile, int(time.time()), role, content))
    conn.commit(); conn.close()

def load_history(mobile: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT role, content, ts FROM conversations WHERE mobile=? ORDER BY ts ASC", (mobile,))
    rows = c.fetchall(); conn.close()
    return [{"role": r, "content": t, "ts": ts} for (r, t, ts) in rows if r in ("user","assistant")]

def load_all_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, mobile, ts, role, content FROM conversations ORDER BY ts ASC", conn)
    conn.close()
    return df

init_db()

# -------------------------------
# 2) Google Drive helpers
# -------------------------------
def list_drive_files(folder_id: str, api_key: str) -> List[Dict[str,Any]]:
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,mimeType)",
        "key": api_key,
        "pageSize": 2000,
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    return r.json().get("files", [])

@st.cache_data(show_spinner=False)
def download_drive_file_bytes(file_id: str, api_key: str) -> bytes:
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    r = requests.get(url, params={"alt":"media","key":api_key}, timeout=120)
    r.raise_for_status()
    return r.content

# -------------------------------
# 3) Shard mapping & loader (small meta reads only for UI)
# -------------------------------
def slugify(name: str) -> str:
    # consistent with builder: safe name derived from filename without extension
    return re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(name)[0])[:80]

@st.cache_data(show_spinner=False)
def map_drive_shards(folder_id: str, api_key: str) -> Dict[str, Dict[str,Any]]:
    """
    Builds a mapping: display_name -> {meta_id, emb_id, meta_name, emb_name}
    display_name is the original PDF filename (used for UI labels).
    meta_ and embeddings_ files must be uploaded by the sharded builder.
    """
    files = list_drive_files(folder_id, api_key)
    mapping: Dict[str, Dict[str,Any]] = {}
    # first collect ids by file name pattern
    for f in files:
        nm = f.get("name","")
        if nm.startswith("meta_") and nm.endswith(".json"):
            key = nm[len("meta_"):-len(".json")]
            mapping.setdefault(key, {})["meta_id"] = f["id"]
            mapping[key]["meta_name"] = nm
        elif nm.startswith("embeddings_") and (nm.endswith(".npy.gz") or nm.endswith(".npy")):
            key = nm[len("embeddings_"):]
            key = key.replace(".npy.gz","").replace(".npy","")
            mapping.setdefault(key, {})["emb_id"] = f["id"]
            mapping[key]["emb_name"] = nm

    # Try to recover friendly display name by reading meta files (small)
    resolved: Dict[str, Dict[str,Any]] = {}
    for key, val in mapping.items():
        display = key
        if "meta_id" in val:
            try:
                b = download_drive_file_bytes(val["meta_id"], api_key)
                meta = json.loads(b.decode("utf-8"))
                # meta is a list of chunks; meta[0]["file"] should be original PDF filename
                if isinstance(meta, list) and len(meta) and "file" in meta[0]:
                    display = meta[0]["file"]
            except Exception:
                pass
        resolved[display] = val
        resolved[display]["display_name"] = display
    return resolved

# -------------------------------
# 4) Build Chroma from selected shards (lazy)
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_chroma_from_selected(selected_tuple: Tuple[str,...], shard_map: Dict[str,Any], api_key: str):
    """
    selected_tuple: tuple of display names (strings). shard_map is result of map_drive_shards.
    Returns (chroma_collection, combined_meta_list)
    """
    if not selected_tuple:
        return None, None

    cclient = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
    coll = cclient.create_collection(name=f"manekrit_dynamic_{int(time.time())}", embedding_function=None)

    combined_meta: List[Dict[str,Any]] = []
    for display_name in selected_tuple:
        entry = shard_map.get(display_name)
        if not entry:
            # skip if missing
            continue
        # download meta (small)
        try:
            meta_bytes = download_drive_file_bytes(entry["meta_id"], api_key)
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception as e:
            st.warning(f"Could not download meta for {display_name}: {e}")
            continue

        # download embeddings (gz or raw)
        try:
            emb_bytes = download_drive_file_bytes(entry["emb_id"], api_key)
        except Exception as e:
            st.warning(f"Could not download embeddings for {display_name}: {e}")
            continue

        if entry.get("emb_name","").endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(emb_bytes), mode="rb") as gz:
                embeddings = np.load(gz)
        else:
            embeddings = np.load(io.BytesIO(emb_bytes))

        docs = [m["text"] for m in meta]
        metadatas = [{"project": m["file"], "chunk_id": m["chunk_id"]} for m in meta]
        ids = [f"{slugify(display_name)}__{i}" for i in range(len(meta))]

        # Add to Chroma collection
        coll.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())

        combined_meta.extend(meta)

    return coll, combined_meta

# -------------------------------
# 5) Retrieval + answer generation
# -------------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def retrieve_filtered(query: str, coll, meta: List[Dict[str,Any]], k: int, allowed_projects: Set[str] | None):
    q = embed_query(query)
    CANDIDATES = max(50, k * 10)
    res = coll.query(query_embeddings=[q.tolist()], n_results=CANDIDATES)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    hits = []
    if allowed_projects and "NONE_OF_THE_ABOVE" not in allowed_projects:
        for d, m in zip(docs, metas):
            if m.get("project") in allowed_projects:
                hits.append({"project": m.get("project"), "chunk_id": m.get("chunk_id"), "text": d})
                if len(hits) >= k:
                    break
        if len(hits) < k:
            hits = []
            for d, m in list(zip(docs, metas))[:k]:
                hits.append({"project": m.get("project"), "chunk_id": m.get("chunk_id"), "text": d})
    else:
        for d, m in list(zip(docs, metas))[:k]:
            hits.append({"project": m.get("project"), "chunk_id": m.get("chunk_id"), "text": d})
    return hits

FORMAT_GUIDE = """
Short answer
- 1–2 line direct answer.

Key facts (if available)
- Bullet points (names, dates, approvals, numbers).

My recommendation
- Practical next steps for the buyer.
---
Ask another question or refine (location, budget, legal, pricing).
""".strip()

def answer_from_context(user_query: str, context_chunks: List[Dict[str,Any]]) -> str:
    context_texts = [c["text"] for c in context_chunks]
    system = (
        "You are a Bengaluru real estate analyst. Use ONLY the provided context; do not invent facts. "
        "Be concise at top, then expand with bullets and recommendations."
    )
    user = (
        f"User query: {user_query}\n\nContext snippets:\n"
        + "\n\n---\n\n".join(context_texts)
        + "\n\nFormatting instructions:\n" + FORMAT_GUIDE
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content

# -------------------------------
# 6) UI — ManeKrit
# -------------------------------
st.set_page_config(page_title="ManeKrit", page_icon="🏠", layout="wide")
# hide streamlit chrome for embedded look
st.markdown("""<style>#MainMenu{visibility:hidden} header{visibility:hidden} footer{visibility:hidden}</style>""", unsafe_allow_html=True)

st.markdown("<h1 style='margin-bottom:0.2rem;'>🏠 ManeKrit</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#666;margin-bottom:1.2rem;'>Smart real estate guidance for Bengaluru buyers.</div>", unsafe_allow_html=True)

# session defaults
if "authed" not in st.session_state: st.session_state.authed = False
if "mobile" not in st.session_state: st.session_state.mobile = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "selected_projects" not in st.session_state: st.session_state.selected_projects = set()
if "project_confirmed" not in st.session_state: st.session_state.project_confirmed = False
if "project_map" not in st.session_state: st.session_state.project_map = {}
if "project_list_display" not in st.session_state: st.session_state.project_list_display = []
if "is_admin" not in st.session_state: st.session_state.is_admin = False
if "kb_ready" not in st.session_state: st.session_state.kb_ready = False
if "coll" not in st.session_state: st.session_state.coll = None
if "meta" not in st.session_state: st.session_state.meta = None

def is_valid_mobile(x: str) -> bool:
    return bool(re.match(r"^\+?\d{10,13}$", x.strip()))

# Login + admin expander
if not st.session_state.authed and not st.session_state.is_admin:
    with st.container():
        st.markdown("### Sign in to continue")
        mobile = st.text_input("Mobile", placeholder="+91XXXXXXXXXX", key="login_mobile")
        if st.button("Continue", key="login_mobile_btn", use_container_width=True):
            if not is_valid_mobile(mobile):
                st.error("Enter a valid mobile (10-13 digits).")
            else:
                st.session_state.authed = True
                st.session_state.mobile = mobile.strip()
                st.session_state.chat_history = load_history(st.session_state.mobile)
                log_event(st.session_state.mobile, "event", "login")
                st.rerun()

    with st.expander("Admin login"):
        pwd = st.text_input("Admin password", type="password")
        if st.button("Login as Admin", key="admin_login_btn"):
            if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Invalid admin password.")
    st.stop()

# Admin view
if st.session_state.is_admin:
    tab1, tab2 = st.tabs(["📊 Admin Dashboard","🗂 Conversation Explorer"])
    with tab1:
        df = load_all_df()
        if df.empty:
            st.info("No data yet.")
        else:
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="s") + pd.to_timedelta(TZ_OFFSET, unit="h")
            df["date"] = df["ts_dt"].dt.date
            messages = len(df[df["role"].isin(["user","assistant"])])
            users = df["mobile"].nunique()
            dau = df.groupby("date")["mobile"].nunique().iloc[-1] if not df.empty else 0
            avg_msgs = round(messages / users, 2) if users else 0.0
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total messages", f"{messages}")
            c2.metric("Total users", f"{users}")
            c3.metric("DAU (today)", f"{dau}")
            c4.metric("Avg msgs / user", f"{avg_msgs}")
            daily_msgs = df[df["role"].isin(["user","assistant"])].groupby("date").size().reset_index(name="messages")
            st.markdown("#### Messages per day"); st.line_chart(daily_msgs.set_index("date"))
            ev = df[df["role"]=="event"]["content"].fillna("")
            sel = ev[ev.str.startswith("project_selection:")]
            if not sel.empty:
                proj_counts={}
                for row in sel:
                    names_str = row.split("project_selection:",1)[1].strip()
                    for nm in [x.strip() for x in names_str.split(",") if x.strip()]:
                        proj_counts[nm] = proj_counts.get(nm,0)+1
                if proj_counts:
                    proj_df = pd.DataFrame(sorted(proj_counts.items(), key=lambda x: x[1], reverse=True), columns=["project","selections"])
                    st.markdown("#### Top selected projects"); st.bar_chart(proj_df.set_index("project"))
    with tab2:
        df = load_all_df()
        if df.empty:
            st.info("No conversations yet.")
        else:
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="s") + pd.to_timedelta(TZ_OFFSET, unit="h")
            cols = st.columns(3)
            with cols[0]:
                mobiles = ["(all)"] + sorted(df["mobile"].unique().tolist())
                f_mobile = st.selectbox("Mobile", mobiles, index=0)
            with cols[1]:
                min_d, max_d = df["ts_dt"].dt.date.min(), df["ts_dt"].dt.date.max()
                f_from = st.date_input("From", value=min_d)
            with cols[2]:
                min_d, max_d = df["ts_dt"].dt.date.min(), df["ts_dt"].dt.date.max()
                f_to = st.date_input("To", value=max_d)
            mask = (df["ts_dt"].dt.date >= f_from) & (df["ts_dt"].dt.date <= f_to)
            if f_mobile != "(all)":
                mask &= (df["mobile"] == f_mobile)
            fdf = df[mask].sort_values("ts_dt")[["ts_dt","mobile","role","content"]]
            st.markdown("#### Messages"); st.dataframe(fdf, use_container_width=True, hide_index=True)
            csv = fdf.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="manekrit_conversations.csv", mime="text/csv")
    st.stop()

# ---------- User flow (sharded) ----------
# Required config check
missing=[]
if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
if not GDRIVE_API_KEY: missing.append("GDRIVE_API_KEY")
if not GDRIVE_FOLDER_ID: missing.append("GDRIVE_FOLDER_ID")
if missing:
    st.error("Configuration missing. Please set: " + ", ".join(missing))
    st.stop()

# Build shard map (small)
try:
    shard_map = map_drive_shards(GDRIVE_FOLDER_ID, GDRIVE_API_KEY)
    # project display names for UI
    display_projects = sorted([k for k in shard_map.keys()])
    st.session_state.project_list_display = display_projects
except Exception as e:
    st.error(f"Could not list files from Drive: {e}")
    st.stop()

# Project picker UI (no heavy downloads)
if not st.session_state.project_confirmed:
    st.markdown("### Which projects are you considering right now?")
    st.caption("Select one or more, or choose None of the above to ask general questions.")
    if not st.session_state.project_list_display:
        st.info("No projects found in Drive. Ensure meta_*.json files are uploaded.")
        st.stop()

    cols = st.columns(3)
    for i, name in enumerate(st.session_state.project_list_display):
        disp = name
        sel = name in st.session_state.selected_projects
        label = f"✅ {disp}" if sel else disp
        if cols[i % 3].button(label, key=f"proj_{i}", use_container_width=True):
            if sel:
                st.session_state.selected_projects.remove(name)
            else:
                if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects:
                    st.session_state.selected_projects.remove("NONE_OF_THE_ABOVE")
                st.session_state.selected_projects.add(name)

    st.divider()
    c1, c2 = st.columns([1,1])
    with c1:
        none_sel = "NONE_OF_THE_ABOVE" in st.session_state.selected_projects
        none_label = "✅ None of the above" if none_sel else "None of the above"
        if st.button(none_label, key="proj_none", use_container_width=True):
            if none_sel:
                st.session_state.selected_projects.remove("NONE_OF_THE_ABOVE")
            else:
                st.session_state.selected_projects = {"NONE_OF_THE_ABOVE"}
    with c2:
        if st.button("Continue", key="confirm_projects", use_container_width=True):
            # build Chroma only for selected projects (lazy)
            chosen = ("All projects" if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects
                      else ", ".join(sorted(st.session_state.selected_projects)))
            log_event(st.session_state.mobile, "event", f"project_selection: {chosen}")
            # prepare selected tuple of display names
            if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects:
                # load all shards (may be heavy)
                selected = tuple(st.session_state.project_list_display)
            else:
                selected = tuple(sorted(st.session_state.selected_projects))
            with st.spinner("Preparing knowledge base for your selection…"):
                coll, meta = build_chroma_from_selected(selected, shard_map, GDRIVE_API_KEY)
            if coll is None or meta is None:
                st.error("Failed to load selected projects. Check shards in Drive.")
                st.stop()
            st.session_state.coll = coll
            st.session_state.meta = meta
            st.session_state.kb_ready = True
            st.session_state.project_confirmed = True
            st.rerun()
    st.stop()

# After confirmation: ensure KB loaded
if not st.session_state.kb_ready or st.session_state.coll is None:
    st.error("Knowledge base not loaded. Click Change selection and Continue to load it.")
    st.stop()

coll = st.session_state.coll
meta = st.session_state.meta

# Topbar with selection summary + Change selection
left, right = st.columns([0.7, 0.3])
with left:
    if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects:
        st.caption("Project filter: **All projects**")
    else:
        st.caption("Project filter: **" + ", ".join(sorted(st.session_state.selected_projects)) + "**")
with right:
    if st.button("Change selection", key="change_selection_btn"):
        st.session_state.project_confirmed = False
        st.session_state.kb_ready = False
        st.session_state.coll = None
        st.session_state.meta = None
        st.rerun()

st.subheader("Chat")

# Render past messages
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    elif chat["role"] == "assistant":
        st.markdown(f"**Advisor:** {chat['content']}")

# Input form
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Type your question here…", key="user_q_input")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not user_q.strip():
        st.warning("Enter a question.")
    else:
        log_event(st.session_state.mobile, "user", user_q)
        with st.spinner("Thinking…"):
            allowed = None
            if st.session_state.selected_projects and "NONE_OF_THE_ABOVE" not in st.session_state.selected_projects:
                allowed = set(st.session_state.selected_projects)
            top = retrieve_filtered(user_q, coll, meta, TOP_K, allowed)
            ans = answer_from_context(user_q, top)
        log_event(st.session_state.mobile, "assistant", ans)
        st.session_state.chat_history.append({"role":"user","content":user_q,"ts":int(time.time())})
        st.session_state.chat_history.append({"role":"assistant","content":ans,"ts":int(time.time())})
        st.rerun()
