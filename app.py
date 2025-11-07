# app.py — MaaneKrit (Path A: prebuilt index from Drive; ultra-fast startup)
import os
import io
import re
import json
import time
import hashlib
import requests
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# -------------------------------
# 0) CONFIG / SECRETS (env first, then Streamlit secrets if deployed)
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

CHAT_MODEL       = "gpt-5"
EMBED_MODEL      = "text-embedding-3-small"  # matches builder for best results
TOP_K            = 5

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# 1) GOOGLE DRIVE HELPERS
# -------------------------------
def list_drive_files(folder_id: str, api_key: str) -> List[Dict[str, str]]:
    """List files (id, name, md5Checksum, modifiedTime) in a Drive folder."""
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

def find_by_name(files: List[Dict[str, str]], name: str):
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

# -------------------------------
# 2) LOAD PREBUILT INDEX
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_prebuilt_index(folder_id: str, api_key: str):
    """Download index.faiss + meta.json from Drive and load them."""
    files = list_drive_files(folder_id, api_key)
    idx_f = find_by_name(files, "index.faiss")
    meta_f = find_by_name(files, "meta.json")
    if not idx_f or not meta_f:
        return None, None  # missing prebuilt files

    idx_bytes  = download_drive_file_bytes(idx_f["id"], api_key)
    meta_bytes = download_drive_file_bytes(meta_f["id"], api_key)

    # Save index bytes to a temp file (faiss.read_index needs a path)
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, "index.faiss")
    with open(idx_path, "wb") as fp:
        fp.write(idx_bytes)

    index = faiss.read_index(idx_path)
    meta  = json.loads(meta_bytes.decode("utf-8"))
    return index, meta

# -------------------------------
# 3) RETRIEVAL + ANSWERING
# -------------------------------
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def retrieve(query: str, index, meta, k: int = TOP_K) -> List[Dict[str, Any]]:
    q = embed_query(query)
    D, I = index.search(np.array([q]), k)
    out = []
    for idx in I[0]:
        if idx >= 0:
            m = meta[idx]
            out.append({"project": m["file"], "chunk_id": m["chunk_id"], "text": m["text"]})
    return out

FORMAT_GUIDE = """
Respond like this:

Short answer
- A crisp 1–2 line direct answer.

Key facts (only if available in context)
- Bullet points of specific details (names, numbers, dates, approvals, survey numbers, timelines).

My recommendation
- Practical next steps and checks a buyer should do based on the context.

After that, ALWAYS end with this:

---
Let me know your next question — location, budget, legal check, pricing, anything.
Type your next query below:
""".strip()

def answer_from_context(user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_texts = [c["text"] for c in context_chunks]
    system = (
        "You are a Bengaluru real estate analyst. "
        "Answer ONLY using the provided context. "
        "If the context is insufficient, explicitly say what's missing and what to ask for. "
        "Do not reveal or mention sources, filenames, chunk IDs, or citations. "
        "Be concise upfront, then detailed. Use bullet points where helpful. "
        "Follow the exact format instructions provided."
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
# 4) UI — Avaasa Krit (clean, fast)
# -------------------------------
st.set_page_config(page_title="MaaneKrit", page_icon="🏠", layout="wide")
st.markdown("<h1 style='margin-bottom:0.2rem;'>🏠 Maane Krit</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#666;margin-bottom:1.2rem;'>Smart real estate guidance for Bengaluru buyers.</div>", unsafe_allow_html=True)

# Lightweight login (email or mobile)
if "authed" not in st.session_state:
    st.session_state.authed = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def is_valid_email(x: str) -> bool:
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", x.strip()))

def is_valid_mobile(x: str) -> bool:
    return bool(re.match(r"^\+?\d{10,13}$", x.strip()))

if not st.session_state.authed:
    with st.container(border=True):
        st.markdown("### Sign in to continue")
        tab1, tab2 = st.tabs(["Email", "Mobile"])
        with tab1:
            email = st.text_input("Email", placeholder="you@example.com", key="login_email")
            if st.button("Continue", key="login_email_btn", type="primary", use_container_width=True):
                if not is_valid_email(email):
                    st.error("Please enter a valid email.")
                else:
                    st.session_state.authed = True
                    st.session_state.user_id = email.strip().lower()
                    st.rerun()
        with tab2:
            mobile = st.text_input("Mobile", placeholder="+91XXXXXXXXXX", key="login_mobile")
            if st.button("Continue ", key="login_mobile_btn", type="primary", use_container_width=True):
                if not is_valid_mobile(mobile):
                    st.error("Please enter a valid mobile number (10–13 digits).")
                else:
                    st.session_state.authed = True
                    st.session_state.user_id = mobile.strip()
                    st.rerun()
    st.stop()

# Guard config
missing = []
if not OPENAI_API_KEY:   missing.append("OPENAI_API_KEY")
if not GDRIVE_API_KEY:   missing.append("GDRIVE_API_KEY")
if not GDRIVE_FOLDER_ID: missing.append("GDRIVE_FOLDER_ID")
if missing:
    st.error("Configuration missing. Please set: " + ", ".join(missing))
    st.stop()

# Load prebuilt index (fast path)
with st.spinner("Preparing knowledge base…"):
    index, meta = load_prebuilt_index(GDRIVE_FOLDER_ID, GDRIVE_API_KEY)

if index is None or meta is None:
    st.error("Prebuilt index not found. Upload 'index.faiss' and 'meta.json' to your Drive folder.")
    st.stop()

# Chat UI
st.subheader("Chat")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Advisor:** {chat['bot']}")

with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Type your question here…", key="user_q_input")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not user_q.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Thinking…"):
            top = retrieve(user_q, index, meta, TOP_K)
            ans = answer_from_context(user_q, top)
        st.session_state.chat_history.append({"user": user_q, "bot": ans})
        st.rerun()
