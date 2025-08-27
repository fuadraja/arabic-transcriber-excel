import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import tempfile
from mutagen import File as MutagenFile
from openai import OpenAI

st.set_page_config(page_title="Arabic Transcriber → Excel", page_icon="📝", layout="wide")
st.title("📝 Arabic Transcriber → Excel")

# --- API key from Streamlit Secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.warning("Add OPENAI_API_KEY in App → Settings → Secrets before running.", icon="⚠️")

# Sidebar: self-fill defaults for all rows
st.sidebar.header("Defaults (apply to all rows)")
project = st.sidebar.text_input("Project", value=st.session_state.get("project", "My Project"))
client = st.sidebar.text_input("Client", value=st.session_state.get("client", ""))
speaker = st.sidebar.text_input("Speaker / Agent", value=st.session_state.get("speaker", ""))
notes = st.sidebar.text_area("Notes (applied to all)", value=st.session_state.get("notes", ""))

extra_help = "Optional lines like: Key=Value\nExample:\nCallType=Inbound\nPriority=Normal"
extra_kv = st.sidebar.text_area("Extra fields (key=value per line)", value="CallType=Inbound\nPriority=Normal", help=extra_help)

def parse_kv(text: str):
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data

extra_fields = parse_kv(extra_kv)

# Transcription settings
st.sidebar.header("Transcription")
model = st.sidebar.selectbox("Model", ["whisper-1", "gpt-4o-mini-transcribe"], index=0, help="Whisper-1 is recommended.")
language = st.sidebar.selectbox("Recording language", ["ar", "auto"], index=0)

st.markdown("**Upload one or more audio files** (mp3, m4a, wav, ogg, webm).")
uploaded_files = st.file_uploader("Audio files", type=["mp3", "m4a", "wav", "ogg", "webm"], accept_multiple_files=True)

default_xlsx = f"transcripts_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
xlsx_name = st.text_input("Excel output filename", value=default_xlsx)
go = st.button("🚀 Transcribe")

# persist sidebar inputs
st.session_state["project"] = project
st.session_state["client"] = client
st.session_state["speaker"] = speaker
st.session_state["notes"] = notes

def detect_duration_seconds(tmp_path: str):
    try:
        m = MutagenFile(tmp_path)
        if m is not None and getattr(m, "info", None) and getattr(m.info, "length", None):
            return round(float(m.info.length), 2)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transcripts")
    return bio.getvalue()

if go:
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY secret.", icon="❌")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one audio file.", icon="📎")
        st.stop()

    client_api = OpenAI(api_key=OPENAI_API_KEY)
    rows = []
    progress = st.progress(0.0)
    status = st.empty()

    for idx, f in enumerate(uploaded_files, start=1):
        status.info(f"Transcribing: {f.name}")

        suffix = ""
        if "." in f.name:
            suffix = "." + f.name.rsplit(".", 1)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name

        duration = detect_duration_seconds(tmp_path)

        try:
            kwargs = {
                "model": model,
                "file": open(tmp_path, "rb"),
                "response_format": "text",
            }
            if language != "auto":
                kwargs["language"] = "ar"

            text_result = client_api.audio.transcriptions.create(**kwargs)
            transcript_text = text_result if isinstance(text_result, str) else getattr(text_result, "text", "")
        except Exception as e:
            transcript_text = f"[ERROR] {type(e).__name__}: {e}"

        row = {
            "Project": project,
            "Client": client,
            "Speaker": speaker,
            "RecordingDate": datetime.today().date().isoformat(),
            "FileName": f.name,
            "DurationSec": duration,
            "Language": "ar" if language != "auto" else "auto",
            "Transcript": transcript_text,
            "Notes": notes,
        }
        row.update(extra_fields)
        rows.append(row)
        progress.progress(idx / len(uploaded_files))

    status.success("Done ✔️")
    df = pd.DataFrame(rows)
    st.subheader("Preview")
    st.dataframe(df, use_container_width=True, height=400)

    excel_bytes = to_excel_bytes(df)
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=xlsx_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
