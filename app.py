# app.py
# ManeKrit — Embed existing HTML visually inside Streamlit + full Streamlit UI below
#
# How to use:
# 1. Place this file as app.py in your repo.
# 2. Ensure requirements.txt includes: streamlit, openai, chromadb, requests, numpy==1.26.4, pandas, python-dotenv
# 3. Set Streamlit secrets (Manage app -> Secrets) or environment variables:
#    OPENAI_API_KEY, GDRIVE_API_KEY, GDRIVE_FOLDER_ID, ADMIN_PASSWORD (optional), SHARD_BASE_URL (optional)
# 4. Push to GitHub & redeploy streamlit cloud or run locally with `streamlit run app.py`.
#
# Notes:
# - The top of the page renders your original HTML (unchanged). That is static/visual.
# - The interactive flows are implemented using Streamlit widgets below the embedded HTML.
# - The embedded HTML's Try Now buttons call `scrollTo` to bring the user to the interactive Streamlit area.
# - If you'd prefer full JS ↔ Python messaging, we can add it, but it's more fragile; ask me when ready.

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os, re, io, json, time, gzip, sqlite3, requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Set
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Config helpers
# -------------------------
def get_cfg(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY   = get_cfg("OPENAI_API_KEY")
GDRIVE_API_KEY   = get_cfg("GDRIVE_API_KEY")
GDRIVE_FOLDER_ID = get_cfg("GDRIVE_FOLDER_ID")
SHARD_BASE_URL   = get_cfg("SHARD_BASE_URL", "").rstrip("/")
ADMIN_PASSWORD   = get_cfg("ADMIN_PASSWORD", "adminpass")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-5"
TOP_K       = int(get_cfg("TOP_K", "3"))
MAX_DOWNLOAD_WORKERS = int(get_cfg("MAX_DOWNLOAD_WORKERS", "4"))

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Insert your full HTML/CSS/JS here (the HTML you gave earlier).
# We will embed it into the Streamlit page for exact visuals.
# To keep the code readable, I put the HTML into a Python triple-quoted string.
# If the HTML is large, you may want to load it from a separate file.
# -------------------------

EMBEDDED_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ManeKrit | Bengaluru Real Estate Advisor</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
    <script src="https://cdn.jsdelivr.net/npm/feather-icons/dist/feather.min.js"></script>
    <style>
    /* Minimal inline CSS adjustments for embedded mode */
    body { margin: 0; font-family: Inter, system-ui, sans-serif; }
    .embedded-container { padding: 32px; background: linear-gradient(90deg,#1e3a8a,#2563eb); color: white; }
    .try-btn { background: white; color: #1e3a8a; }
    /* Ensure the hero section is well-contained when embedded in Streamlit */
    .hero { padding: 48px 24px; }
    </style>
</head>
<body>
  <div class="embedded-container">
    <div class="hero max-w-4xl mx-auto text-center">
      <div class="inline-flex items-center justify-center bg-white/10 rounded-full px-4 py-1 text-sm tracking-wide mb-4">
        <span class="font-semibold">ManeKrit</span>
      </div>
      <h1 class="text-4xl sm:text-5xl font-bold leading-tight mb-4">Your Trusted Guide in Bengaluru's Real Estate Journey</h1>
      <p class="text-xl text-blue-100 mb-6">We've helped navigate over 1,00,000 sq.ft. of Bengaluru's property landscape. Let us guide your home buying journey with expert insights.</p>
      <div class="flex justify-center gap-3">
        <button id="embedded-try-now" class="try-btn px-6 py-3 rounded-lg font-medium">Try Now</button>
        <a href="#services" class="border border-white/70 hover:bg-white/10 rounded-lg px-6 py-3 transition duration-300">Learn More</a>
      </div>
    </div>
  </div>

  <!-- Some trust cards (visual) -->
  <section class="py-12 bg-white">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div class="text-center p-6 bg-blue-50 rounded-xl">
          <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 text-blue-800 mb-4">
            <i data-feather="home"></i>
          </div>
          <h3 class="text-xl font-semibold text-gray-800">1,00,000+ sq.ft.</h3>
          <p class="text-gray-600">Transaction Experience</p>
        </div>
        <div class="text-center p-6 bg-blue-50 rounded-xl">
          <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 text-blue-800 mb-4">
            <i data-feather="users"></i>
          </div>
          <h3 class="text-xl font-semibold text-gray-800">100+</h3>
          <p class="text-gray-600">Families Guided</p>
        </div>
        <div class="text-center p-6 bg-blue-50 rounded-xl">
          <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 text-blue-800 mb-4">
            <i data-feather="map-pin"></i>
          </div>
          <h3 class="text-xl font-semibold text-gray-800">Bengaluru</h3>
          <p class="text-gray-600">Deep Local Expertise</p>
        </div>
      </div>
    </div>
  </section>

  <!-- CTA -->
  <section id="services" class="py-16 bg-gray-50">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <h2 class="text-3xl font-bold text-gray-800 mb-4">Closing the Information Gap in Bengaluru Real Estate</h2>
      <p class="mt-4 text-lg text-gray-600 max-w-3xl mx-auto">First-time home buyers face an overwhelming amount of conflicting information. We bring clarity.</p>
    </div>
  </section>

  <script>
    feather.replace();
    // When user clicks embedded Try Now, scroll the parent window to the interactive Streamlit area
    document.getElementById('embedded-try-now').addEventListener('click', function() {
      // scroll the parent (Streamlit page) to anchor #streamlit-app-interactive
      // If parent is not same-origin, this will still work for top-level navigation
      try {
        if (window.parent) {
          // find anchor element in parent and scroll to it
          // We'll change parent.location.hash to highlight it
          window.parent.location.hash = 'streamlit-app-interactive';
        } else {
          // fallback: open the same page with hash
          window.location.hash = 'streamlit-app-interactive';
        }
      } catch (e) {
        // last fallback: open current window's hash
        window.location.hash = 'streamlit-app-interactive';
      }
    });
  </script>
</body>
</html>
"""

# -------------------------
# Below the embedded HTML we render the fully-functional Streamlit UI.
# The interactive region has the id "streamlit-app-interactive" so the embedded HTML can scroll to it.
# -------------------------

# Top-level page config
st.set_page_config(page_title="ManeKrit", page_icon="🏠", layout="wide")
# Hide streamlit default chrome (optional)
st.markdown("<style>#MainMenu{visibility:hidden} footer{visibility:hidden} header{visibility:hidden}</style>", unsafe_allow_html=True)

# Render the embedded HTML visually
st.components.v1.html(EMBEDDED_HTML, height=700, scrolling=True)

# Add a named anchor so embedded HTML's button can scroll to this area
st.markdown("<div id='streamlit-app-interactive'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------
# Now the full Streamlit interactive app (login, project picker, sharded loader, chat)
# This is essentially the same functional Streamlit app we designed previously.
# For brevity I include a compact but complete version that handles:
# - mobile login
# - project picker using shards (CDN or Drive)
# - parallel downloads of selected shards, building Chroma collection
# - chat input + retrieval + answer via GPT
# - sqlite history and admin dashboard
# -----------------------------------------------------------------------------------

# --- SQLite initializer & helpers
DB_PATH = "conversations.sqlite"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mobile TEXT NOT NULL, ts INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_conv_mobile_ts ON conversations (mobile, ts)")
    conn.commit(); conn.close()

def log_event(mobile: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO conversations (mobile, ts, role, content) VALUES (?, ?, ?, ?)",
              (mobile, int(time.time()), role, content))
    conn.commit(); conn.close()

def load_history(mobile: str):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT role, content, ts FROM conversations WHERE mobile=? ORDER BY ts ASC", (mobile,))
    rows = c.fetchall(); conn.close()
    return [{"role": r, "content": t, "ts": ts} for (r, t, ts) in rows]

init_db()

# --- Drive + CDN helpers (small)
def list_drive_files(folder_id: str, api_key: str):
    url = "https://www.googleapis.com/drive/v3/files"
    params = {"q": f"'{folder_id}' in parents and trashed=false",
              "fields":"files(id,name,mimeType)", "key":api_key, "pageSize":2000,
              "supportsAllDrives":True, "includeItemsFromAllDrives":True}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    return r.json().get("files", [])

def download_drive_file_bytes(file_id: str, api_key: str) -> bytes:
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    r = requests.get(url, params={"alt":"media","key":api_key}, timeout=120); r.raise_for_status()
    return r.content

def slugify(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(name)[0])[:80]

@st.cache_data(show_spinner=False)
def map_shards_from_drive(folder_id: str, api_key: str):
    files = list_drive_files(folder_id, api_key)
    mapping = {}
    for f in files:
        nm = f.get("name","")
        if nm.startswith("meta_") and nm.endswith(".json"):
            key = nm[len("meta_"):-len(".json")]
            mapping.setdefault(key, {})["meta_id"] = f["id"]; mapping[key]["meta_name"]=nm
        if nm.startswith("embeddings_") and (nm.endswith(".npy.gz") or nm.endswith(".npy")):
            key = nm[len("embeddings_"):]; key = key.replace(".npy.gz","").replace(".npy","")
            mapping.setdefault(key, {})["emb_id"] = f["id"]; mapping[key]["emb_name"]=nm
    # resolve display names by peeking at meta
    resolved={}
    for key, val in mapping.items():
        display = key
        if "meta_id" in val:
            try:
                b = download_drive_file_bytes(val["meta_id"], api_key); meta = json.loads(b.decode("utf-8"))
                if isinstance(meta, list) and meta and "file" in meta[0]:
                    display = meta[0]["file"]
            except Exception:
                pass
        resolved[display] = val; resolved[display]["display_name"]=display
    return resolved

# CDN index loader (optional)
@st.cache_data(show_spinner=False)
def load_shard_index_from_cdn(shard_base_url: str):
    idx_url = shard_base_url + "/shard_index.json"
    r = requests.get(idx_url, timeout=20); r.raise_for_status()
    return r.json()

# Parallel shard downloader
def download_shard_files_for_display(display_name: str, entry: Dict[str,Any], api_key: str, shard_base_url: str):
    if shard_base_url:
        safe = slugify(display_name)
        meta_url = f"{shard_base_url}/meta_{safe}.json"; emb_url = f"{shard_base_url}/embeddings_{safe}.npy.gz"
        try:
            rb = requests.get(meta_url, timeout=30); rb.raise_for_status()
            meta = json.loads(rb.content.decode("utf-8"))
            remb = requests.get(emb_url, timeout=60); remb.raise_for_status()
            with gzip.GzipFile(fileobj=io.BytesIO(remb.content), mode="rb") as gz:
                embeddings = np.load(gz)
            return display_name, meta, embeddings
        except Exception:
            pass
    # Drive fallback
    if not entry:
        return display_name, None, None
    try:
        meta_bytes = download_drive_file_bytes(entry["meta_id"], api_key)
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception:
        return display_name, None, None
    try:
        emb_bytes = download_drive_file_bytes(entry["emb_id"], api_key)
        if entry.get("emb_name","").endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(emb_bytes), mode="rb") as gz:
                embeddings = np.load(gz)
        else:
            embeddings = np.load(io.BytesIO(emb_bytes))
    except Exception:
        return display_name, meta, None
    return display_name, meta, embeddings

# Build chroma from selected (parallel)
@st.cache_resource(show_spinner=False)
def build_chroma_for_selected(selected: Tuple[str,...], shard_map: Dict[str,Any], api_key: str, shard_base_url: str, progress_callback=None):
    if not selected:
        return None, None
    cclient = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
    coll = cclient.create_collection(name=f"manekrit_session_{int(time.time())}", embedding_function=None)
    combined_meta=[]
    total=len(selected)
    futures=[]
    with ThreadPoolExecutor(max_workers=min(MAX_DOWNLOAD_WORKERS, max(1,total))) as ex:
        for display in selected:
            entry = shard_map.get(display)
            futures.append(ex.submit(download_shard_files_for_display, display, entry, api_key, shard_base_url))
        done=0
        for fut in as_completed(futures):
            display_name, meta, embeddings = fut.result()
            done+=1
            if progress_callback:
                try: progress_callback(done/total)
                except Exception: pass
            if not meta or embeddings is None: continue
            docs=[m["text"] for m in meta]
            metadatas=[{"project": m["file"], "chunk_id": m["chunk_id"]} for m in meta]
            ids=[f"{slugify(display_name)}__{i}" for i in range(len(meta))]
            coll.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())
            combined_meta.extend(meta)
    return coll, combined_meta

# Embedding & retrieval helpers
EMBED_CACHE={}
def embed_query(text: str):
    key = text.strip()[:512]
    if key in EMBED_CACHE: return EMBED_CACHE[key]
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype="float32")
    EMBED_CACHE[key] = vec
    return vec

def retrieve_filtered(query: str, coll, meta, k: int, allowed_projects: Optional[Set[str]]=None):
    q = embed_query(query)
    CANDIDATES = max(30, k * 8)
    res = coll.query(query_embeddings=[q.tolist()], n_results=CANDIDATES)
    docs = res.get("documents", [[]])[0]; metas = res.get("metadatas", [[]])[0]
    hits=[]
    if allowed_projects and "NONE_OF_THE_ABOVE" not in allowed_projects:
        for d,m in zip(docs, metas):
            if m.get("project") in allowed_projects:
                hits.append({"project": m.get("project"), "chunk_id": m.get("chunk_id"), "text": d})
                if len(hits)>=k: break
        if len(hits)<k:
            hits=[]
            for d,m in list(zip(docs, metas))[:k]:
                hits.append({"project": m.get("project"), "chunk_id": m.get("chunk_id"), "text": d})
    else:
        for d,m in list(zip(docs, metas))[:k]:
            hits.append({"project": m.get("project"), "chunk_id": m.get("chunk_id"), "text": d})
    return hits

def answer_from_context(user_query: str, context_chunks: List[Dict[str,Any]]):
    context_texts = [c["text"] for c in context_chunks]
    system = "You are a Bengaluru real estate analyst. Use ONLY the provided context. Be concise then provide bullets and recommendations."
    user = f"User query: {user_query}\n\nContext:\n" + "\n\n---\n\n".join(context_texts)
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return resp.choices[0].message.content

# -------------------------
# Streamlit interactive UI (login, project pick, build & chat)
# -------------------------
# session defaults
defaults = {"authed":False,"mobile":None,"chat_history":[],"selected_projects":set(),"project_confirmed":False,
            "project_map":{},"project_list_display":[],"kb_ready":False,"coll":None,"meta":None,"is_admin":False}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

def is_valid_mobile(x: str): return bool(re.match(r"^\+?\d{10,13}$", x.strip()))

# Login
if not st.session_state.authed and not st.session_state.is_admin:
    st.subheader("Sign in to try the ManeKrit Advisor")
    mobile = st.text_input("Mobile (with country code)", key="mobile_input")
    if st.button("Continue", key="login_continue"):
        if not is_valid_mobile(mobile or ""):
            st.error("Enter valid mobile (10-13 digits).")
        else:
            st.session_state.authed=True; st.session_state.mobile=mobile.strip()
            st.session_state.chat_history = load_history(st.session_state.mobile)
            log_event(st.session_state.mobile, "event", "login")
            st.experimental_rerun()
    with st.expander("Admin login"):
        pwd = st.text_input("Admin password", type="password")
        if st.button("Login as Admin", key="admin_btn"):
            if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                st.session_state.is_admin=True; st.experimental_rerun()
            else:
                st.error("Invalid admin password.")
    st.stop()

# Admin
if st.session_state.is_admin:
    st.header("Admin Dashboard")
    df = pd.read_sql_query("SELECT id, mobile, ts, role, content FROM conversations ORDER BY ts DESC", sqlite3.connect(DB_PATH))
    st.dataframe(df.head(200))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "conversations.csv", "text/csv")
    if st.button("Logout admin"):
        st.session_state.is_admin=False; st.experimental_rerun()
    st.stop()

# Prepare project list (CDN or Drive)
shard_map={}
if SHARD_BASE_URL:
    try:
        shard_map_cdn = load_shard_index_from_cdn(SHARD_BASE_URL)
        # expected format: display_name -> {"meta_url":..., "emb_url":...}
        for disp,v in shard_map_cdn.items():
            shard_map[disp] = {"meta_name": v.get("meta_name", f"meta_{slugify(disp)}.json"),
                               "emb_name": v.get("emb_name", f"embeddings_{slugify(disp)}.npy.gz"),
                               "meta_url": v.get("meta_url"), "emb_url": v.get("emb_url")}
    except Exception:
        try:
            shard_map = map_shards_from_drive(GDRIVE_FOLDER_ID, GDRIVE_API_KEY)
        except Exception as e:
            st.error(f"Cannot list shards: {e}"); st.stop()
else:
    try:
        shard_map = map_shards_from_drive(GDRIVE_FOLDER_ID, GDRIVE_API_KEY)
    except Exception as e:
        st.error(f"Cannot list shards: {e}"); st.stop()

display_projects = sorted(shard_map.keys()); st.session_state.project_list_display=display_projects; st.session_state.project_map={d:d for d in display_projects}

# Project picker
if not st.session_state.project_confirmed:
    st.markdown("### Which projects are you considering right now?")
    st.caption("Select one or more, or choose None of the above.")
    cols = st.columns(3)
    for i, disp in enumerate(display_projects):
        sel = disp in st.session_state.selected_projects
        label = f"✅ {disp}" if sel else disp
        if cols[i%3].button(label, key=f"proj_{i}"):
            if sel: st.session_state.selected_projects.remove(disp)
            else:
                if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects:
                    st.session_state.selected_projects.remove("NONE_OF_THE_ABOVE")
                st.session_state.selected_projects.add(disp)
    st.divider()
    c1,c2 = st.columns([1,1])
    with c1:
        none_sel = "NONE_OF_THE_ABOVE" in st.session_state.selected_projects
        none_label = "✅ None of the above" if none_sel else "None of the above"
        if st.button(none_label, key="none_pick"):
            if none_sel: st.session_state.selected_projects.remove("NONE_OF_THE_ABOVE")
            else: st.session_state.selected_projects = {"NONE_OF_THE_ABOVE"}
    with c2:
        if st.button("Continue", key="pick_continue"):
            chosen = ("All projects" if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects
                      else ", ".join(sorted(st.session_state.selected_projects)))
            log_event(st.session_state.mobile, "event", f"project_selection: {chosen}")
            if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects:
                selected_tuple = tuple(display_projects)
            else:
                selected_tuple = tuple(sorted(st.session_state.selected_projects))
            pb = st.progress(0.0)
            def cb(frac): 
                try: pb.progress(min(1.0, frac))
                except: pass
            with st.spinner("Preparing knowledge base for your selection..."):
                coll, meta = build_chroma_for_selected(selected_tuple, shard_map, GDRIVE_API_KEY, SHARD_BASE_URL, progress_callback=cb)
            pb.progress(1.0)
            if coll is None or not meta:
                st.error("Failed to load selected projects. Check shard files and permissions.")
                st.stop()
            st.session_state.coll=coll; st.session_state.meta=meta; st.session_state.kb_ready=True; st.session_state.project_confirmed=True
            st.experimental_rerun()
    st.stop()

# After confirmation
if not st.session_state.kb_ready or st.session_state.coll is None:
    st.error("Knowledge base not loaded. Click 'Change selection' to reload.")
    st.stop()

coll = st.session_state.coll; meta = st.session_state.meta

left, right = st.columns([0.7,0.3])
with left:
    if "NONE_OF_THE_ABOVE" in st.session_state.selected_projects or not st.session_state.selected_projects:
        st.caption("Project filter: **All projects**")
    else:
        st.caption("Project filter: **" + ", ".join(sorted(st.session_state.selected_projects)) + "**")
with right:
    if st.button("Change selection", key="change_sel"):
        st.session_state.project_confirmed=False; st.session_state.kb_ready=False; st.session_state.coll=None; st.session_state.meta=None; st.experimental_rerun()

st.subheader("Ask the Advisor")
for msg in st.session_state.chat_history:
    if msg["role"]=="user": st.markdown(f"**You:** {msg['content']}")
    else: st.markdown(f"**Advisor:** {msg['content']}")

with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Type your question (location, budget, legal check, pricing...)")
    submitted = st.form_submit_button("Ask")
if submitted:
    if not user_q or not user_q.strip(): st.warning("Write a question.")
    else:
        log_event(st.session_state.mobile, "user", user_q)
        with st.spinner("Thinking..."):
            allowed = None
            if st.session_state.selected_projects and "NONE_OF_THE_ABOVE" not in st.session_state.selected_projects:
                allowed = set(st.session_state.selected_projects)
            top_chunks = retrieve_filtered(user_q, coll, meta, TOP_K, allowed)
            ans = answer_from_context(user_q, top_chunks)
        log_event(st.session_state.mobile, "assistant", ans)
        st.session_state.chat_history.append({"role":"user","content":user_q,"ts":int(time.time())})
        st.session_state.chat_history.append({"role":"assistant","content":ans,"ts":int(time.time())})
        st.experimental_rerun()

# Done
