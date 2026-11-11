"""
AI Text Assistant — OpenRouter + Streamlit
One API key gives access to hundreds of models (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen ...).
Run:  streamlit run openrouter_app.py
Key:  set OPENROUTER_API_KEY as an environment variable, or put it in .streamlit/secrets.toml
"""

import os

import requests
import streamlit as st
from openai import OpenAI

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SYSTEM = "You are a helpful, concise assistant for engineering students."
MAX_TURNS = 20  # keep last N user+assistant pairs to limit tokens (and cost)

st.set_page_config(page_title="AI Assistant (OpenRouter)", page_icon="🤖")
st.title("🤖 AI Assistant — OpenRouter")


# ---------- API key ----------
def get_api_key():
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass  # no secrets.toml file — fall back to the environment variable
    return os.getenv("OPENROUTER_API_KEY", "")


api_key = get_api_key()
if not api_key:
    st.error("No API key found. Set OPENROUTER_API_KEY (see README) and restart the app.")
    st.stop()

client = OpenAI(base_url=BASE_URL, api_key=api_key)


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
    st.header("Settings")
    free_only = st.checkbox("Free models only", value=True,
                            help="Free models have daily request limits; paid models use your credits.")
    try:
        models = [m for m in fetch_models() if m["free"] or not free_only]
        labels = {m["id"]: m["name"] for m in models}
        model = st.selectbox("Model", list(labels), format_func=lambda i: labels[i])
    except Exception as e:
        st.warning(f"Couldn't load the model list ({e}). Type a model ID instead.")
        model = st.text_input("Model ID", "openrouter/auto")
    st.caption(f"`{model}`")

    system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM, height=120)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max reply length (tokens)", 256, 4096, 1024, 256)

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()


# ---------- Chat history ----------
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- New message ----------
def stream_reply(messages):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_headers={"X-Title": "AI Text Assistant"},  # app name shown on your OpenRouter dashboard
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Type your message..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": "system", "content": system_prompt}] + st.session_state.history[-MAX_TURNS * 2:]

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(messages))
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                msg = "Rate limit reached for this model. Wait a minute, or pick another model."
            elif "401" in msg:
                msg = "Invalid API key. Check OPENROUTER_API_KEY."
            elif "402" in msg:
                msg = "Out of credits for this paid model. Pick a free model or add credits."
            reply = f"⚠️ {msg}"
            st.error(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})