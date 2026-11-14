"""
AI Text Assistant — OpenRouter + Streamlit
Run:  streamlit run OR_Chatbot.py
Key:  OPENROUTER_API_KEY as an environment variable, or in .streamlit/secrets.toml (Streamlit Cloud: Secrets)
"""

import json
import os
import uuid
from datetime import datetime

import requests
import streamlit as st
from openai import APIConnectionError, APIStatusError, OpenAI

BASE_URL = "https://openrouter.ai/api/v1"
AUTO_MODEL = "openrouter/free"        # OpenRouter's free router: picks any available free model
SYSTEM_PROMPT = "You are a helpful, concise assistant for engineering students."  # hidden from the UI
MAX_TURNS = 20                        # past user+assistant pairs sent with each request
HISTORY_FILE = "chat_history.json"    # where chats are saved between sessions


st.set_page_config(page_title="AI Assistant", page_icon="🤖")


# ---------- API key ----------
def get_secret(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # no secrets.toml — fall back to environment variable
    return os.getenv(name, default)


api_key = get_secret("OPENROUTER_API_KEY")
if not api_key:
    st.error("No API key found. Set OPENROUTER_API_KEY and restart the app.")
    st.stop()

# On a public deployment, set SAVE_CHATS = "false" in Secrets: a saved file would be shared by all visitors.
SAVE_TO_DISK = get_secret("SAVE_CHATS", "true").lower() == "true"
client = OpenAI(base_url=BASE_URL, api_key=api_key)


# ---------- Chat history storage ----------
def load_chats():
    if SAVE_TO_DISK and os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_chats():
    if SAVE_TO_DISK:
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chats, f, ensure_ascii=False, indent=1)
        except OSError:
            pass


def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "created": datetime.now().isoformat(timespec="seconds"),
                                   "messages": []}
    st.session_state.current = cid


if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if not st.session_state.get("current") or st.session_state.current not in st.session_state.chats:
    # reopen the newest empty chat if there is one, otherwise start a new chat
    empty = [c for c, v in st.session_state.chats.items() if not v["messages"]]
    if empty:
        st.session_state.current = empty[-1]
    else:
        new_chat()

chat = st.session_state.chats[st.session_state.current]


# ---------- Model list (fetched live, cached for 1 hour) ----------
@st.cache_data(ttl=3600)
def fetch_models():
    r = requests.get(f"{BASE_URL}/models", timeout=20)
    r.raise_for_status()
    models = []
    for m in r.json()["data"]:
        p = m.get("pricing", {})
        free = m["id"].endswith(":free") or (p.get("prompt") == "0" and p.get("completion") == "0")
        models.append({"id": m["id"], "name": m.get("name", m["id"]), "free": free})
    return sorted(models, key=lambda x: x["name"].lower())


# ---------- Sidebar ----------
with st.sidebar:
    if st.button("➕ New chat", type="primary"):
        new_chat()
        st.rerun()

    st.subheader("Model")
    free_only = st.checkbox("Free models only", value=True)
    labels = {AUTO_MODEL: "⭐ Auto — best available free model"}
    try:
        labels.update({m["id"]: m["name"] for m in fetch_models() if m["free"] or not free_only})
    except Exception:
        st.caption("Couldn't load the model list; using Auto.")
    model = st.selectbox("Model", list(labels), format_func=lambda i: labels[i], label_visibility="collapsed")

    with st.expander("⚙️ Advanced settings"):
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_tokens = st.slider("Max reply length (tokens)", 512, 8192, 2048, 256,
                               help="Reasoning models use part of this for hidden 'thinking'.")

    st.subheader("🕘 Chat history")
    for cid, c in sorted(st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True):
        if not c["messages"] and cid != st.session_state.current:
            continue
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(c["title"][:32], key=f"open_{cid}",
                         type="primary" if cid == st.session_state.current else "secondary"):
                st.session_state.current = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}", help="Delete this chat"):
                del st.session_state.chats[cid]
                st.session_state.current = None
                save_chats()
                st.rerun()

    if chat["messages"]:
        md = "\n\n".join(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}"
                         for m in chat["messages"])
        st.download_button("⬇️ Download this chat", md, file_name=f"{chat['title'][:30]}.md")


# ---------- Model call with automatic fallback ----------
def build_messages(history):
    """Last MAX_TURNS exchanges; system prompt merged into the first user message
    (some free models reject the 'system' role)."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-MAX_TURNS * 2:]]
    if msgs and msgs[0]["role"] == "user":
        msgs[0]["content"] = f"{SYSTEM_PROMPT}\n\n{msgs[0]['content']}"
    else:
        msgs.insert(0, {"role": "user", "content": SYSTEM_PROMPT})
    return msgs


def models_to_try(selected):
    """Chosen model first; if it's busy or rate-limited, fall back to the free router (3 tries)."""
    return [selected] + [AUTO_MODEL] * (3 if selected != AUTO_MODEL else 2)


def stream_reply(messages, info):
    errors = []
    for m in models_to_try(model):
        try:
            stream = client.chat.completions.create(
                model=m, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True,
                extra_headers={"X-Title": "AI Text Assistant"},
                # keep hidden reasoning short so tokens are left for the answer
                extra_body={"reasoning": {"effort": "low", "exclude": True}},
            )
            got_text = False
            for chunk in stream:
                if getattr(chunk, "model", None):
                    info["model"] = chunk.model
                if chunk.choices and chunk.choices[0].delta.content:
                    got_text = True
                    yield chunk.choices[0].delta.content
            if got_text:
                return
            errors.append(f"{m}: empty reply")
        except APIStatusError as e:
            if e.status_code == 401:
                raise RuntimeError("Invalid API key. Check OPENROUTER_API_KEY.")
            if e.status_code == 402:
                errors.append(f"{m}: no credits")
            else:
                errors.append(f"{m}: HTTP {e.status_code}")
        except APIConnectionError:
            raise RuntimeError("Can't reach openrouter.ai. Check your internet connection.")
    raise RuntimeError("All models are busy right now. Please try again in a minute. (" + "; ".join(errors) + ")")


# ---------- Main chat area ----------
st.title("🤖 AI Assistant")

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("model"):
            st.caption(f"Answered by {msg['model']}")

if prompt := st.chat_input("Type your message..."):
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "New chat":
        chat["title"] = prompt.strip().replace("\n", " ")[:40]
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        info = {}
        try:
            reply = st.write_stream(stream_reply(build_messages(chat["messages"]), info))
            used = info.get("model", model)
            st.caption(f"Answered by {used}")
            chat["messages"].append({"role": "assistant", "content": reply, "model": used})
            save_chats()
            st.rerun()  # refresh the sidebar so the new chat title appears in the history list
        except Exception as e:
            st.error(f"⚠️ {e}")
            chat["messages"].pop()  # don't keep the failed question in history; just send it again
            if not chat["messages"]:
                chat["title"] = "New chat"
    save_chats()