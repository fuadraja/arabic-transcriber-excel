import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import mimetypes
import os

# اختياري: تحويل تلقائي عند فشل الإرسال بصيغ معيّنة
# يتطلّب وجود ffmpeg في النظام (نوفّره عبر packages.txt)
from pydub import AudioSegment

from openai import OpenAI

# ================= إعدادات الواجهة =================
st.set_page_config(page_title="تفريغ المقابلات إلى إكسل", page_icon="📝", layout="wide")
st.title("📝 تفريغ المقابلات الصوتية (عربي) → إكسل")
st.caption("ارفع تسجيلات صوتية بالعربية. سنفرّغها نصيًا ونحفظ النتائج في ملف إكسل بالأعمدة المطلوبة.")

# ================= مفتاح OpenAI =================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
if not OPENAI_API_KEY:
    with st.expander("🔐 أدخل مفتاح OpenAI API (اختياري إذا لم تستطع استخدام Secrets)"):
        OPENAI_API_KEY = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            placeholder="sk-********************************",
            help="يُستخدم فقط في جلستك الحالية ولن يُحفظ.",
        )
if not OPENAI_API_KEY:
    st.info("يفضّل إضافة المفتاح من Settings → Secrets في Streamlit Cloud. أو أدخله مؤقتًا أعلاه.", icon="🔑")

# ================= الإعدادات (الشريط الجانبي) =================
with st.sidebar:
    st.header("⚙️ الإعدادات")
    model = st.selectbox(
        "نموذج التفريغ",
        options=[
            "gpt-4o-transcribe",   # جرّبه أولًا
            "whisper-1",           # بديل ثابت
        ],
        index=0,
        help="إذا لم يعمل الأول على حسابك جرّب whisper-1."
    )
    temperature = st.slider("Temperature (اختياري)", 0.0, 1.0, 0.0, 0.1)
    st.caption("اتركها 0 للحصول على نص أدقّ.")

# ================= دوال مساعدة =================
ALLOWED_EXTS = {"flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"}

def _guess_mime(filename: str, fallback: str = None) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if not mime and fallback:
        return fallback
    return mime or "application/octet-stream"

def _file_ext(filename: str) -> str:
    return (os.path.splitext(filename)[1] or "").lower().lstrip(".")

def _convert_to_mp3_in_memory(file_bytes: bytes, src_ext: str) -> tuple[bytes, str, str]:
    """
    يحاول تحويل الملف إلى MP3 داخل الذاكرة باستخدام pydub/ffmpeg.
    يعيد (mp3_bytes, new_name_extension, mime)
    """
    audio = AudioSegment.from_file(BytesIO(file_bytes), format=src_ext if src_ext else None)
    mp3_buf = BytesIO()
    audio.export(mp3_buf, format="mp3", bitrate="192k")
    mp3_buf.seek(0)
    return mp3_buf.read(), "mp3", "audio/mpeg"

def transcribe_once(client: OpenAI, file_name: str, file_bytes: bytes, mime: str) -> str:
    """
    محاولة واحدة للتفريغ عبر واجهة OpenAI (تُعيد نصًا).
    """
    result = client.audio.transcriptions.create(
        model=model,
        file=(file_name, file_bytes, mime),
        language="ar",
        response_format="text",
        temperature=temperature,
    )
    if isinstance(result, str):
        return result.strip()
    return getattr(result, "text", "").strip()

def transcribe_with_fallback(client: OpenAI, st_file) -> str:
    """
    يحاول إرسال الملف بصيغته الأصلية مع تمرير MIME الصحيح.
    إذا فشل بخطأ 'Invalid file format'، يحاول تحويله إلى MP3 وإعادة المحاولة.
    """
    # اقرأ البايتات واحفظ المؤشر
    raw = st_file.read()
    st_file.seek(0)

    name = st_file.name
    ext = _file_ext(name)
    if ext not in ALLOWED_EXTS:
        raise ValueError(
            f"الامتداد ({ext}) غير مدعوم. الامتدادات المسموحة: {sorted(ALLOWED_EXTS)}"
        )

    mime = _guess_mime(name, fallback=f"audio/{ext if ext!='mpga' else 'mpeg'}")
    try:
        return transcribe_once(client, name, raw, mime)
    except Exception as e:
        msg = str(e)
        # إذا كانت المشكلة في الصيغة، جرّب تحويل MP3
        keywords = ["Invalid file format", "invalid_request_error", "unsupported", "format"]
        if any(k.lower() in msg.lower() for k in keywords):
            try:
                mp3_bytes, new_ext, new_mime = _convert_to_mp3_in_memory(raw, ext)
                new_name = f"{os.path.splitext(name)[0]}.{new_ext}"
                return transcribe_once(client, new_name, mp3_bytes, new_mime)
            except Exception as conv_err:
                raise RuntimeError(f"فشل التحويل إلى MP3 بعد رفض الصيغة الأصلية: {conv_err}") from e
        # أخطاء أخرى: أعِدها كما هي
        raise

# ================= نموذج إدخال البيانات =================
st.markdown("### ✍️ بيانات كل ملف")
with st.form("meta_form", clear_on_submit=False):
    st.markdown("#### 🗂️ ارفع عدّة ملفات (يمكن لكل ملف بيانات مختلفة)")
    uploaded_files = st.file_uploader(
        "ارفع ملفات الصوت (MP3/WAV/M4A/MP4/…)",
        type=list(ALLOWED_EXTS),
        accept_multiple_files=True
    )

    st.divider()
    st.markdown("#### 🧩 تعبئة بيانات افتراضية (يمكن تعديلها لكل ملف)")
    col1, col2, col3 = st.columns(3)
    with col1:
        default_company = st.text_input("اسم الشركة (افتراضي)")
        default_employee = st.text_input("اسم الموظف (افتراضي)")
    with col2:
        default_job = st.text_input("الوظيفة (افتراضي)")
        default_exp = st.text_input("الخبرة (افتراضي)", placeholder="مثال: 5 سنوات")
    with col3:
        default_spec = st.text_input("الاختصاص (افتراضي)")
        use_defaults = st.checkbox("استخدام القيم الافتراضية تلقائيًا", value=True)

    # حقول لكل ملف
    per_file_meta = []
    if uploaded_files:
        st.markdown("#### 📝 بيانات مخصّصة لكل ملف")
        for idx, f in enumerate(uploaded_files, start=1):
            with st.expander(f"الملف #{idx}: {f.name}", expanded=not use_defaults):
                c1, c2, c3 = st.columns(3)
                with c1:
                    company = st.text_input(f"اسم الشركة - {f.name}", value=default_company if use_defaults else "", key=f"company_{idx}")
                    employee = st.text_input(f"اسم الموظف - {f.name}", value=default_employee if use_defaults else "", key=f"employee_{idx}")
                with c2:
                    job = st.text_input(f"الوظيفة - {f.name}", value=default_job if use_defaults else "", key=f"job_{idx}")
                    exp = st.text_input(f"الخبرة - {f.name}", value=default_exp if use_defaults else "", key=f"exp_{idx}")
                with c3:
                    spec = st.text_input(f"الاختصاص - {f.name}", value=default_spec if use_defaults else "", key=f"spec_{idx}")
                per_file_meta.append({
                    "company": company,
                    "employee": employee,
                    "job": job,
                    "exp": exp,
                    "spec": spec,
                })

    submit = st.form_submit_button("بدء التفريغ ▶️")

# ================= التنفيذ =================
if submit:
    if not OPENAI_API_KEY:
        st.error("الرجاء إدخال مفتاح OpenAI API أولًا.", icon="🚫")
        st.stop()
    if not uploaded_files:
        st.warning("الرجاء رفع ملف صوتي واحد على الأقل.", icon="📎")
        st.stop()

    client = OpenAI(api_key=OPENAI_API_KEY)

    rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded_files, start=1):
        status.info(f"جارٍ تفريغ: {f.name} ...")
        # اجلب البيانات الخاصة بهذا الملف
        meta = per_file_meta[i-1] if i-1 < len(per_file_meta) else {
            "company": default_company, "employee": default_employee,
            "job": default_job, "exp": default_exp, "spec": default_spec
        }
        try:
            text = transcribe_with_fallback(client, f)
        except Exception as e:
            text = f"خطأ أثناء التفريغ: {e}"

        rows.append({
            "اسم الشركة": meta.get("company", "") or "",
            "اسم الموظف": meta.get("employee", "") or "",
            "الوظيفة": meta.get("job", "") or "",
            "الخبرة": meta.get("exp", "") or "",
            "الاختصاص": meta.get("spec", "") or "",
            "المقابلة (النص المفرغ)": text or "",
        })
        progress.progress(i / len(uploaded_files))

    status.success("اكتمل التفريغ ✅")

    # DataFrame + معاينة
    df = pd.DataFrame(rows, columns=[
        "اسم الشركة", "اسم الموظف", "الوظيفة", "الخبرة", "الاختصاص", "المقابلة (النص المفرغ)"
    ])

    st.subheader("📄 المعاينة")
    st.dataframe(df, use_container_width=True)

    # Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="النتائج")
    excel_buffer.seek(0)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_name = f"تفريغ_المقابلات_{now}.xlsx"

    st.download_button(
        label="⬇️ تحميل إكسل",
        data=excel_buffer,
        file_name=excel_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # CSV
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ تحميل CSV (UTF-8)",
        data=csv_data,
        file_name=f"تفريغ_المقابلات_{now}.csv",
        mime="text/csv"
    )

# ================= ملاحظات =================
with st.expander("💡 ملاحظات هامة"):
    st.markdown(
        """
- يمرّر التطبيق نوع الملف (MIME) تلقائيًا. إذا رفضت الواجهة الصيغة الأصلية، سيحوّل الملف إلى MP3 تلقائيًا ويعيد المحاولة.
- إذا لم يعمل **gpt-4o-transcribe**، جرّب **whisper-1** من الشريط الجانبي.
- لا تحفظ مفتاحك داخل الكود أو GitHub. استخدم **Secrets** أو الحقل المؤقت داخل التطبيق.
        """
    )
