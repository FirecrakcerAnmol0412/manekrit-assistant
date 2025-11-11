# app.py — ManeKrit (mobile login + per-number history + project picker + Path A index + Admin & Analytics)
import os
import re
import io
import json
import time
import sqlite3
import requests
import numpy as np
import faiss_cpu as faiss
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Set, Optional

# -------------------------------
# 0) CONFIG / SECRETS
# -------------------------------
load_dotenv()

def get_config(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return default

OPENAI_API_KEY   = get_config("OPENAI_API_KEY")
GDRIVE_API_KEY   = get_config("GDRIVE_API_KEY")
GDRIVE_FOLDER_ID = get_config("GDRIVE_FOLDER_ID")
ADMIN_PASSWORD   = get_config("ADMIN_PASSWORD")  # <-- set this in Secrets/.env

CHAT_MODEL  = "gpt-5"
EMBED_MODEL = "text-embedding-3-small"   # must match your prebuilt index
TOP_K       = 5
TZ_OFFSET   = 5.5  # IST for simple charts; for precise tz use pytz

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# 1) PERSISTENCE (SQLite)
# -------------------------------
DB_PATH = "conversations.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mobile TEXT NOT NULL,
            ts INTEGER NOT NULL,
            role TEXT NOT NULL,      -- 'system' | 'user' | 'assistant' | 'event'
            content TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_conv_mobile_ts ON conversations (mobile, ts)")
    conn.commit()
    conn.close()

def log_event(mobile: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (mobile, ts, role, content) VALUES (?, ?, ?, ?)",
        (mobile, int(time.time()), role, content)
    )
    conn.commit()
    conn.close()

def load_history(mobile: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content, ts FROM conversations WHERE mobile=? ORDER BY ts ASC", (mobile,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": t, "ts": ts} for (r, t, ts) in rows if r in ("user", "assistant")]

def load_all_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, mobile, ts, role, content FROM conversations ORDER BY ts ASC", conn)
    conn.close()
    return df

init_db()

# -------------------------------
# 2) GOOGLE DRIVE HELPERS (Path A)
# -------------------------------
def list_drive_files(folder_id: str, api_key: str) -> List[Dict[str, str]]:
    url = "https://www.googleapis.com/drive/v3/files"
    q = f"'{folder_id}' in parents and trashed=false"
    params = {
        "q": q,
        "fields": "files(id,name,md5Checksum,modifiedTime,mimeType)",
        "key": api_key,
        "pageSize": 1000,
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("files", [])

def find_by_name(files: List[Dict[str, str]], name: str) -> Optional[Dict[str, str]]:
    for f in files:
        if f.get("name") == name:
            return f
    return None

@st.cache_data(show_spinner=False)
def download_drive_file_bytes(file_id: str, api_key: str) -> bytes:
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    r = requests.get(url, params={"alt":"media","key":api_key}, timeout=120)
    r.raise_for_status()
    return r.content

@st.cache_resource(show_spinner=False)
def load_prebuilt_index(folder_id: str, api_key: str):
    files = list_drive_files(folder_id, api_key)
    idx_f = find_by_name(files, "index.faiss")
    meta_f = find_by_name(files, "meta.json")
    if not idx_f or not meta_f:
        return None, None
    idx_bytes  = download_drive_file_bytes(idx_f["id"], api_key)
    meta_bytes = download_drive_file_bytes(meta_f["id"], api_key)
    cache_dir = ".cache"; os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, "index.faiss")
    with open(idx_path, "wb") as fp: fp.write(idx_bytes)
    index = faiss.read_index(idx_path)
    meta  = json.loads(meta_bytes.decode("utf-8"))
    return index, meta

# -------------------------------
# 3) RETRIEVAL + ANSWERING
# -------------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def retrieve_filtered(query: str, index, meta, k: int, allowed_projects: Set[str] | None):
    q = embed_query(query)
    CANDIDATES = max(50, k * 10)
    D, I = index.search(np.array([q]), CANDIDATES)
    hits = []
    if allowed_projects and "NONE_OF_THE_ABOVE" not in allowed_projects:
        for idx in I[0]:
            if idx < 0: continue
            m = meta[idx]
            if m["file"] in allowed_projects:
                hits.append({"project": m["file"], "chunk_id": m["chunk_id"], "text": m["text"]})
            if len(hits) >= k: break
        if len(hits) < k:
            hits = []
            for idx in I[0][:k]:
                if idx < 0: continue
                m = meta[idx]
                hits.append({"project": m["file"], "chunk_id": m["chunk_id"], "text": m["text"]})
    else:
        for idx in I[0][:k]:
            if idx < 0: continue
            m = meta[idx]
            hits.append({"project": m["file"], "chunk_id": m["chunk_id"], "text": m["text"]})
    return hits

FORMAT_GUIDE = """
Short answer
- A crisp 1–2 line direct answer.

Key facts (only if available in context)
- Bullet points of specific details (names, numbers, dates, approvals, survey numbers, timelines).

My recommendation
- Practical next steps and checks a buyer should do based on the context.

---
Let me know your next question — location, budget, legal check, pricing, anything.
Type your next query below:
""".strip()

def answer_from_context(user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_texts = [c["text"] for c in context_chunks]
    system = (
        "You are a Bengaluru real estate analyst. Answer ONLY from the provided context. "
        "If insufficient, say what's missing. Do not reveal sources. Be concise upfront, then detailed."
    )
    user = (
        f"User query: {user_query}\n\n"
        "Context (snippets from project PDFs):\n"
        + "\n\n---\n\n".join(context_texts)
        + "\n\nFormatting instructions:\n"
        + FORMAT_GUIDE
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content

# -------------------------------
# 4) UI — ManeKrit (mobile login → project picker → chat → admin)
# -------------------------------
st.set_page_config(page_title="ManeKrit", page_icon="🏠", layout="wide")
st.markdown("<h1 style='margin-bottom:0.2rem;'>🏠 ManeKrit</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#666;margin-bottom:1.2rem;'>Smart real estate guidance for Bengaluru buyers.</div>", unsafe_allow_html=True)

# Session state
if "authed" not in st.session_state: st.session_state.authed = False
if "mobile" not in st.session_state: st.session_state.mobile = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "selected_projects" not in st.session_state: st.session_state.selected_projects = set()
if "project_confirmed" not in st.session_state: st.session_state.project_confirmed = False
if "project_names" not in st.session_state: st.session_state.project_names = []
if "project_map" not in st.session_state: st.session_state.project_map = {}
if "is_admin" not in st.session_state: st.session_state.is_admin = False

def is_valid_mobile(x: str) -> bool:
    return bool(re.match(r"^\+?\d{10,13}$", x.strip()))

# --- Mobile-only Login (User)
if not st.session_state.authed and not st.session_state.is_admin:
    with st.container(border=True):
        st.markdown("### Sign in to continue")
        mobile = st.text_input("Mobile", placeholder="+91XXXXXXXXXX", key="login_mobile")
        if st.button("Continue", key="login_mobile_btn", type="primary", use_container_width=True):
            if not is_valid_mobile(mobile):
                st.error("Please enter a valid mobile number (10–13 digits).")
            else:
                st.session_state.authed = True
                st.session_state.mobile = mobile.strip()
                st.session_state.chat_history = load_history(st.session_state.mobile)
                log_event(st.session_state.mobile, "event", "login")
                st.rerun()

    # Admin login link
    with st.expander("Admin login"):
        pwd = st.text_input("Admin password", type="password", key="admin_pwd")
        if st.button("Login as Admin", key="admin_login_btn"):
            if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Invalid admin password.")
    st.stop()

# --- Admin panel entry when logged as admin (no user state needed)
if st.session_state.is_admin:
    # simple tabs: Admin Dashboard | Conversation Explorer
    tab1, tab2 = st.tabs(["📊 Admin Dashboard", "🗂️ Conversation Explorer"])

    with tab1:
        df = load_all_df()
        if df.empty:
            st.info("No data yet.")
        else:
            # Prepare time columns
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="s") + pd.to_timedelta(TZ_OFFSET, unit="h")
            df["date"] = df["ts_dt"].dt.date

            # KPIs
            messages = len(df[df["role"].isin(["user", "assistant"])])
            users = df["mobile"].nunique()
            dau = df.groupby("date")["mobile"].nunique().iloc[-1] if not df.empty else 0
            avg_msgs_per_user = round(messages / users, 2) if users else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total messages", f"{messages}")
            c2.metric("Total users", f"{users}")
            c3.metric("DAU (today)", f"{dau}")
            c4.metric("Avg msgs / user", f"{avg_msgs_per_user}")

            # Trends
            daily_msgs = df[df["role"].isin(["user","assistant"])].groupby("date").size().reset_index(name="messages")
            daily_users = df.groupby("date")["mobile"].nunique().reset_index(name="active_users")

            st.markdown("#### Messages per day")
            st.line_chart(daily_msgs.set_index("date"))

            st.markdown("#### Active users per day")
            st.line_chart(daily_users.set_index("date"))

            # Top projects (parse project_selection events)
            ev = df[df["role"] == "event"]["content"].fillna("")
            sel = ev[ev.str.startswith("project_selection:")]
            if not sel.empty:
                # flatten comma-separated names
                proj_counts = {}
                for row in sel:
                    names_str = row.split("project_selection:",1)[1].strip()
                    if names_str.lower().startswith("all projects"):
                        continue
                    for nm in [x.strip() for x in names_str.split(",") if x.strip()]:
                        proj_counts[nm] = proj_counts.get(nm, 0) + 1
                if proj_counts:
                    proj_df = pd.DataFrame(sorted(proj_counts.items(), key=lambda x: x[1], reverse=True),
                                           columns=["project","selections"])
                    st.markdown("#### Top selected projects")
                    st.bar_chart(proj_df.set_index("project"))
                else:
                    st.info("No explicit project selections yet.")
            else:
                st.info("No project selection events yet.")

    with tab2:
        df = load_all_df()
        if df.empty:
            st.info("No conversations yet.")
        else:
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="s") + pd.to_timedelta(TZ_OFFSET, unit="h")
            # Filters
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
            fdf = df[mask].copy()

            st.markdown("#### Messages")
            # A tidy view for review/export
            view = fdf.sort_values("ts_dt")[["ts_dt","mobile","role","content"]]
            st.dataframe(view, use_container_width=True, hide_index=True)

            # Export CSV
            csv = view.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="manekrit_conversations.csv", mime="text/csv")

        st.stop()

# --- From here: regular user experience (already authenticated user) ---
# Guard config
missing = []
if not OPENAI_API_KEY:   missing.append("OPENAI_API_KEY")
if not GDRIVE_API_KEY:   missing.append("GDRIVE_API_KEY")
if not GDRIVE_FOLDER_ID: missing.append("GDRIVE_FOLDER_ID")
if missing:
    st.error("Configuration missing. Please set: " + ", ".join(missing))
    st.stop()

# Load prebuilt index (fast)
with st.spinner("Preparing knowledge base…"):
    index, meta = load_prebuilt_index(GDRIVE_FOLDER_ID, GDRIVE_API_KEY)
if index is None or meta is None:
    st.error("Prebuilt index not found. Upload 'index.faiss' and 'meta.json' to your Drive folder.")
    st.stop()

# Project list/map (strip ".pdf")
if not st.session_state.project_names:
    file_names = sorted({m["file"] for m in meta})
    st.session_state.project_map = {fn: os.path.splitext(fn)[0] for fn in file_names}
    st.session_state.project_names = file_names

# Project picker until confirmed
if not st.session_state.project_confirmed:
    st.markdown("### Which projects are you considering right now?")
    st.caption("Select one or more, or choose **None of the above** to ask general questions.")

    cols = st.columns(3)
    for i, internal_name in enumerate(st.session_state.project_names):
        display_name = st.session_state.project_map.get(internal_name, internal_name)
        is_selected = internal_name in st.session_state.selected_projects
        label = f"✅ {display_name}" if is_selected else display_name
        if cols[i % 3].button(label, key=f"proj_{i}", use_container_width=True):
            if is_selected:
                st.session_state.selected_projects.remove(internal_name)
            else:
                if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects:
                    st.session_state.selected_projects.remove("NONE_OF_THE_ABOVE")
                st.session_state.selected_projects.add(internal_name)

    st.divider()
    c1, c2 = st.columns([1,1])
    with c1:
        none_selected = "NONE_OF_THE_ABOVE" in st.session_state.selected_projects
        none_label = "✅ None of the above" if none_selected else "None of the above"
        if st.button(none_label, key="proj_none", type="secondary", use_container_width=True):
            if none_selected:
                st.session_state.selected_projects.remove("NONE_OF_THE_ABOVE")
            else:
                st.session_state.selected_projects = {"NONE_OF_THE_ABOVE"}
    with c2:
        if st.button("Continue", key="confirm_projects", type="primary", use_container_width=True):
            st.session_state.project_confirmed = True
            chosen = (
                "All projects"
                if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects
                else ", ".join([st.session_state.project_map.get(fn, fn) for fn in sorted(st.session_state.selected_projects)])
            )
            log_event(st.session_state.mobile, "event", f"project_selection: {chosen}")
            st.rerun()
    st.stop()

# ---- Top bar: filter + Change selection ----
left, right = st.columns([0.7, 0.3])
with left:
    if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects:
        st.caption("Project filter: **All projects**")
    else:
        display_list = [st.session_state.project_map.get(fn, fn) for fn in sorted(st.session_state.selected_projects)]
        st.caption("Project filter: **" + ", ".join(display_list) + "**")
with right:
    if st.button("Change selection", key="change_selection_btn"):
        st.session_state.project_confirmed = False
        st.rerun()

st.subheader("Chat")

# Render past messages
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    elif chat["role"] == "assistant":
        st.markdown(f"**Advisor:** {chat['content']}")

# Input
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Type your question here…", key="user_q_input")
    submitted = st.form_submit_button("Ask", type="primary")

# Submit
if submitted:
    if not user_q.strip():
        st.warning("Enter a question.")
    else:
        log_event(st.session_state.mobile, "user", user_q)
        with st.spinner("Thinking…"):
            allowed = None
            if st.session_state.selected_projects and "NONE_OF_THE_ABOVE" not in st.session_state.selected_projects:
                allowed = set(st.session_state.selected_projects)
            top = retrieve_filtered(user_q, index, meta, TOP_K, allowed)
            ans = answer_from_context(user_q, top)
        log_event(st.session_state.mobile, "assistant", ans)
        st.session_state.chat_history.append({"role": "user", "content": user_q, "ts": int(time.time())})
        st.session_state.chat_history.append({"role": "assistant", "content": ans, "ts": int(time.time())})
        st.rerun()

