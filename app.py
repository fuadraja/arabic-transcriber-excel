# app.py
import os
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Tuple
import tempfile

import streamlit as st
import pandas as pd

from openai import OpenAI
from openai import APIError, AuthenticationError, RateLimitError, BadRequestError

# ========================= إعدادات الصفحة =========================
st.set_page_config(page_title="تفريغ المقابلات العربية → إكسل", page_icon="📝", layout="wide")
st.title("📝 تفريغ المقابلات الصوتية (عربي) → إكسل + تحليل نصي (بدون تحويل صيغ)")

st.caption("ارفع ملفات الصوت بصيغة مدعومة، أدخِل بيانات كل ملف (إن رغبت)، سنُفرِّغ النص بالعربية ونُخرج النتائج في ملف إكسل واحد. يمكن أيضًا إجراء تحليل نصي اختياري.")

# ========================= مفاتيح وأمان =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None

if not OPENAI_API_KEY:
    with st.expander("🔐 أدخِل مفتاح OpenAI API (مؤقت في جلستك فقط)"):
        OPENAI_API_KEY = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            placeholder="sk-********************************",
            help="لأفضل أمان استخدم Settings → Secrets على Streamlit Cloud.",
        )

if not OPENAI_API_KEY:
    st.info("لن يعمل التفريغ قبل إدخال مفتاح OpenAI API.", icon="🔑")

# ========================= الشريط الجانبي: الإعدادات =========================
with st.sidebar:
    st.header("⚙️ الإعدادات")

    st.subheader("نموذج التفريغ")
    asr_model = st.selectbox(
        "اختر نموذج التفريغ",
        options=["gpt-4o-transcribe", "whisper-1"],
        index=0,
        help="إن لم يتوفر الأول في حسابك جرّب whisper-1."
    )
    asr_temperature = st.slider("Temperature (تفريغ)", 0.0, 1.0, 0.0, 0.1)

    st.subheader("التحليل النصي (اختياري)")
    enable_nlp = st.checkbox("تفعيل التحليل النصي", value=True, help="ملخص + كلمات مفتاحية + مشاعر + عدد الكلمات.")
    nlp_model = st.selectbox("نموذج التحليل", options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    nlp_depth = st.select_slider("تفصيل الملخص", options=["قصير", "متوسط", "مفصل"], value="متوسط")

    st.subheader("💰 حاسبة التكلفة (تقديري)")
    st.caption("يمكن تعديل الأسعار يدويًا حسب خطة OpenAI لديك.")
    price_per_min_gpt4o_transcribe = st.number_input("سعر/دقيقة - gpt-4o-transcribe ($)", value=0.006, min_value=0.0, step=0.001, format="%.3f")
    price_per_min_whisper = st.number_input("سعر/دقيقة - whisper-1 ($)", value=0.006, min_value=0.0, step=0.001, format="%.3f")
    price_per_1k_tokens_nlp = st.number_input("سعر/1000 توكن للنموذج التحليلي ($)", value=0.002, min_value=0.0, step=0.001, format="%.3f")
    st.caption("ملاحظة: الأسعار هنا افتراضية وقابلة للتعديل لتطابق حسابك.")

# ========================= أدوات مساعدة =========================
SUPPORTED_AUDIO = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']

def _get_file_ext(name: str) -> str:
    return (os.path.splitext(name)[1][1:] or "").lower()

def _safe_duration_seconds(file_name: str, data: bytes) -> float:
    """
    يحاول قراءة مدة الصوت باستخدام mutagen (بدون FFmpeg).
    إن فشل لسبب ما، يعيد -1.
    """
    try:
        from mutagen import File as MutagenFile
        with tempfile.NamedTemporaryFile(delete=True, suffix=f"_{os.path.basename(file_name)}") as tmp:
            tmp.write(data)
            tmp.flush()
            audio = MutagenFile(tmp.name)
            if audio is not None and getattr(audio, "info", None) and getattr(audio.info, "length", None):
                return float(audio.info.length)
        return -1.0
    except Exception:
        return -1.0

def transcribe_bytes(client: OpenAI, name: str, content_bytes: bytes, ext: str, language="ar", temperature=0.0) -> str:
    """
    يرسل الملف إلى واجهة التفريغ ويُعيد النص العربي.
    """
    try:
        resp = client.audio.transcriptions.create(
            model=asr_model,
            file=(f"{os.path.splitext(name)[0]}.{ext}", content_bytes),
            language=language,
            temperature=temperature,
            response_format="text",
        )
        if isinstance(resp, str):
            return resp.strip()
        return getattr(resp, "text", "").strip()
    except AuthenticationError as e:
        raise RuntimeError("فشل التوثيق: تحقّق من مفتاح OpenAI API.") from e
    except RateLimitError as e:
        raise RuntimeError("تجاوزت الحصة/المعدل. راجع خطتك أو أعد المحاولة لاحقًا.") from e
    except BadRequestError as e:
        raise RuntimeError(f"طلب غير صالح: {e}") from e
    except APIError as e:
        raise RuntimeError(f"خطأ من خادم OpenAI: {e}") from e
    except Exception as e:
        raise RuntimeError(f"حدث خطأ أثناء التفريغ: {e}") from e

def analyze_text(client: OpenAI, text: str, depth: str) -> Dict[str, Any]:
    """
    تحليل نصي بسيط:
    - ملخص (حسب العمق)
    - كلمات مفتاحية
    - مشاعر عامة
    - عدد الكلمات
    """
    wc = len(text.split())
    if not text.strip():
        return {"ملخص": "", "كلمات مفتاحية": "", "المشاعر": "", "عدد الكلمات": 0}

    prompt = f"""حلّل النص العربي التالي بإيجاز ووضوح:
- قدّم ملخصًا ({depth}) من 3-6 جمل بحسب العمق.
- استخرج حتى 8 كلمات/عبارات مفتاحية (comma-separated).
- قيّم المشاعر العامة (إيجابي/محايد/سلبي) مع جملة تفسيرية قصيرة.
أعد الإجابة بصيغة JSON بالمفاتيح: summary, keywords, sentiment.
النص:
{text}"""

    try:
        comp = client.chat.completions.create(
            model=nlp_model,
            messages=[
                {"role": "system", "content": "أنت مساعد خبير في تحليل النصوص العربية."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = comp.choices[0].message.content.strip()
        import json, re
        match = re.search(r'\{.*\}', raw, re.S)
        data = {}
        if match:
            data = json.loads(match.group(0))
        else:
            data = {"summary": raw, "keywords": "", "sentiment": ""}

        return {
            "ملخص": data.get("summary", ""),
            "كلمات مفتاحية": data.get("keywords", ""),
            "المشاعر": data.get("sentiment", ""),
            "عدد الكلمات": wc,
        }
    except Exception:
        return {"ملخص": "", "كلمات مفتاحية": "", "المشاعر": "", "عدد الكلمات": wc}

def minutes_from_seconds(sec: float) -> float:
    return max(0.0, round(sec / 60.0, 2)) if sec and sec > 0 else 0.0

# ========================= الإدخال: الملفات + بيانات لكل ملف =========================
with st.form("upload_form", clear_on_submit=False):
    st.subheader("📥 رفع الملفات")
    uploaded_files = st.file_uploader(
        "ارفع ملفات الصوت (يدعم: " + ", ".join(SUPPORTED_AUDIO) + ")",
        type=SUPPORTED_AUDIO,
        accept_multiple_files=True
    )

    st.markdown("—")
    st.subheader("🧾 بيانات عامة (تُستخدم افتراضيًا لكل الملفات)")
    c1, c2, c3 = st.columns(3)
    with c1:
        default_company = st.text_input("اسم الشركة (افتراضي)")
        default_employee = st.text_input("اسم الموظف (افتراضي)")
    with c2:
        default_job = st.text_input("الوظيفة (افتراضي)")
        default_exp = st.text_input("الخبرة (افتراضي)", placeholder="مثال: 5 سنوات")
    with c3:
        default_spec = st.text_input("الاختصاص (افتراضي)")
        per_file_overrides = st.checkbox("سأدخل بيانات مختلفة لكل ملف", value=True)

    per_file_meta: List[Dict[str, str]] = []
    if uploaded_files and per_file_overrides:
        st.markdown("—")
        st.subheader("✍️ بيانات مخصّصة لكل ملف")
        for idx, f in enumerate(uploaded_files, start=1):
            with st.expander(f"الملف #{idx}: {f.name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    comp = st.text_input(f"اسم الشركة #{idx}", key=f"company_{idx}", value=default_company)
                    emp = st.text_input(f"اسم الموظف #{idx}", key=f"employee_{idx}", value=default_employee)
                with col2:
                    job = st.text_input(f"الوظيفة #{idx}", key=f"job_{idx}", value=default_job)
                    exp = st.text_input(f"الخبرة #{idx}", key=f"exp_{idx}", value=default_exp)
                with col3:
                    spec = st.text_input(f"الاختصاص #{idx}", key=f"spec_{idx}", value=default_spec)
                per_file_meta.append({
                    "اسم الشركة": comp, "اسم الموظف": emp, "الوظيفة": job, "الخبرة": exp, "الاختصاص": spec
                })

    submit = st.form_submit_button("▶️ بدء التفريغ")

# ========================= التنفيذ =========================
if submit:
    if not OPENAI_API_KEY:
        st.error("الرجاء إدخال مفتاح OpenAI API أولًا.", icon="🚫")
        st.stop()

    if not uploaded_files:
        st.warning("الرجاء رفع ملف صوتي واحد على الأقل.", icon="📎")
        st.stop()

    # تحقّق مبكر من الامتدادات
    bad_files = [f.name for f in uploaded_files if _get_file_ext(f.name) not in SUPPORTED_AUDIO]
    if bad_files:
        st.error(
            "هذه الملفات غير بصيغة مدعومة، الرجاء تحويلها خارجيًا إلى صيغة مدعومة ثم إعادة الرفع:\n- " + "\n- ".join(bad_files),
            icon="🚫"
        )
        st.stop()

    client = OpenAI(api_key=OPENAI_API_KEY)

    results_rows: List[Dict[str, Any]] = []
    durations_min: List[float] = []

    progress = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded_files, start=1):
        status.info(f"جارٍ معالجة: {f.name} ...")

        # بيانات هذا الملف
        if per_file_overrides and len(per_file_meta) >= i:
            meta = per_file_meta[i-1]
        else:
            meta = {
                "اسم الشركة": default_company,
                "اسم الموظف": default_employee,
                "الوظيفة": default_job,
                "الخبرة": default_exp,
                "الاختصاص": default_spec,
            }

        # اقرأ البايتات واحسب مدة تقريبية
        ext = _get_file_ext(f.name)
        fbytes = f.read()
        f.seek(0)
        duration_sec = _safe_duration_seconds(f.name, fbytes)

        # التفريغ
        try:
            text = transcribe_bytes(client, f.name, fbytes, ext, language="ar", temperature=asr_temperature)
        except Exception as e:
            text = f"خطأ أثناء التفريغ: {e}"

        row = {**meta, "المقابلة (النص المفرغ)": text or ""}

        # التحليل النصي
        if enable_nlp and text and not str(text).startswith("خطأ"):
            analysis = analyze_text(client, text, depth=nlp_depth)
            row.update(analysis)
        elif enable_nlp:
            row.update({"ملخص": "", "كلمات مفتاحية": "", "المشاعر": "", "عدد الكلمات": 0})

        results_rows.append(row)
        durations_min.append(minutes_from_seconds(duration_sec))
        progress.progress(i / len(uploaded_files))

    status.success("اكتمل التفريغ ✅")

    # ========================= عرض النتائج + التحميل =========================
    st.subheader("📄 النتائج")
    cols = ["اسم الشركة", "اسم الموظف", "الوظيفة", "الخبرة", "الاختصاص", "المقابلة (النص المفرغ)"]
    if enable_nlp:
        cols += ["ملخص", "كلمات مفتاحية", "المشاعر", "عدد الكلمات"]

    df = pd.DataFrame(results_rows, columns=cols)
    st.dataframe(df, use_container_width=True, height=420)

    # Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="النتائج")
    excel_buffer.seek(0)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_name = f"تفريغ_المقابلات_{now}.xlsx"

    cdl, cdr = st.columns(2)
    with cdl:
        st.download_button(
            label="⬇️ تحميل إكسل",
            data=excel_buffer,
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with cdr:
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ تحميل CSV (UTF-8)",
            data=csv_data,
            file_name=f"تفريغ_المقابلات_{now}.csv",
            mime="text/csv"
        )

    # ========================= حاسبة التكلفة =========================
    st.subheader("💰 تقدير التكلفة")
    total_minutes = round(sum(durations_min), 2)
    st.caption("إن تعذر استخراج المدة من بعض الملفات ستكون الدقائق 0 لهذه العناصر.")

    if asr_model == "gpt-4o-transcribe":
        asr_cost = total_minutes * float(price_per_min_gpt4o_transcribe)
    else:
        asr_cost = total_minutes * float(price_per_min_whisper)

    total_words = 0
    if enable_nlp:
        try:
            total_words = int(df["عدد الكلمات"].fillna(0).sum())
        except Exception:
            total_words = 0
    est_tokens_nlp = int((total_words * 5) / 4)  # تقدير بسيط
    nlp_cost = (est_tokens_nlp / 1000.0) * float(price_per_1k_tokens_nlp) if enable_nlp else 0.0

    st.write(f"- **إجمالي الدقائق:** ~ {total_minutes} دقيقة")
    st.write(f"- **تكلفة التفريغ (ASR):** ~ ${asr_cost:.4f}")
    if enable_nlp:
        st.write(f"- **تقدير توكنات التحليل:** ~ {est_tokens_nlp} توكن")
        st.write(f"- **تكلفة التحليل (NLP):** ~ ${nlp_cost:.4f}")
    st.markdown(f"**الإجمالي التقديري:** ~ ${asr_cost + nlp_cost:.4f}")

# ========================= ملاحظات =========================
with st.expander("💡 ملاحظات هامة"):
    st.markdown(
        """
- يدعم التطبيق صيغ الصوت: `flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm`.
- لا يوجد تحويل صيغ داخل التطبيق لتجنّب مشاكل البناء على Streamlit Cloud.
- لو ظهرت أخطاء **401** تأكّد من مفتاح OpenAI. لو **429** (حصة/معدل) راجع الفوترة أو أعد المحاولة.
- الأسعار في الحاسبة **تقديرية** ويمكن تعديلها من الشريط الجانبي.
- لا تحفظ مفتاحك داخل الكود أو GitHub. استخدم **Secrets** على Streamlit Cloud أو الحقل المؤقت.
        """
    )
