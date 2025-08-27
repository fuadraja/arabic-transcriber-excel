import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from openai import OpenAI

# ---------------------- إعدادات الصفحة ----------------------
st.set_page_config(page_title="تفريغ المقابلات إلى إكسل", page_icon="📝", layout="wide")
st.title("📝 تفريغ المقابلات الصوتية (عربي) → إكسل")

st.caption("يرجى رفع تسجيلات صوتية بالعربية. سنُفرِّغ النص ونحفظ النتائج في ملف إكسل بالأعمدة المطلوبة.")

# ---------------------- مفتاح OpenAI ----------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None

if not OPENAI_API_KEY:
    with st.expander("🔐 أدخل مفتاح OpenAI API (اختياري إذا لم تستطع استخدام Secrets)"):
        OPENAI_API_KEY = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            placeholder="sk-********************************",
            help="سيُستخدم فقط في جلستك الحالية، ولن يُحفظ على الخادم.",
        )

if not OPENAI_API_KEY:
    st.info("لأفضل أمان، أضِف المفتاح في Settings → Secrets على Streamlit Cloud. أو أدخله مؤقتًا أعلاه.", icon="🔑")

# ---------------------- اختيار نموذج التفريغ ----------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    model = st.selectbox(
        "نموذج التفريغ (يفضّل الأول)",
        options=[
            "gpt-4o-transcribe",  # مخصص للتفريغ
            "whisper-1",          # في حال الأول غير متاح في حسابك
        ],
        index=0
    )
    temperature = st.slider("Temperature (اختياري)", 0.0, 1.0, 0.0, 0.1)
    st.caption("اتركها 0 للحصول على نص أدقّ بدون تنويعات.")

# ---------------------- حقول البيانات (تملأ ذاتياً لكل ملف) ----------------------
with st.form("meta_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        company = st.text_input("اسم الشركة")
        employee = st.text_input("اسم الموظف")
    with col2:
        job_title = st.text_input("الوظيفة")
        experience = st.text_input("الخبرة", placeholder="مثال: 5 سنوات")
    with col3:
        specialization = st.text_input("الاختصاص")
        auto_fill = st.checkbox("استخدام هذه القيم لكل الملفات", value=True)

    uploaded_files = st.file_uploader(
        "ارفع ملفات الصوت (MP3/WAV/M4A/MP4). يمكن اختيار أكثر من ملف",
        type=["mp3", "wav", "m4a", "mp4"],
        accept_multiple_files=True
    )

    submit = st.form_submit_button("بدء التفريغ ▶️")

# ---------------------- الدالة: تفريغ ملف ----------------------
def transcribe_file(client: OpenAI, file):
    """
    يعيد نصًا مُفرّغًا باللغة العربية من ملف صوتي.
    """
    # نمرّر الملف مباشرة دون تحويل
    try:
        # واجهة OpenAI SDK الحديثة
        # gpt-4o-transcribe أو whisper-1
        transcript = client.audio.transcriptions.create(
            model=model,
            file=(file.name, file.read()),
            # التوجيه للغة العربية قد يساعد
            language="ar",
            temperature=temperature,
            response_format="text",
        )
        # transcript يكون نصًا خامًا عند response_format="text"
        if isinstance(transcript, str):
            return transcript.strip()
        # احتياط في بعض الإصدارات
        return getattr(transcript, "text", "").strip()
    finally:
        file.seek(0)  # لإرجاع المؤشر في حال احتجناه لاحقًا

# ---------------------- التنفيذ ----------------------
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
        # نحدد القيم المستخدمة لهذا الملف
        _company = company if auto_fill else st.session_state.get(f"company_{i}", company)
        _employee = employee if auto_fill else st.session_state.get(f"employee_{i}", employee)
        _job = job_title if auto_fill else st.session_state.get(f"job_{i}", job_title)
        _exp = experience if auto_fill else st.session_state.get(f"exp_{i}", experience)
        _spec = specialization if auto_fill else st.session_state.get(f"spec_{i}", specialization)

        try:
            text = transcribe_file(client, f)
        except Exception as e:
            text = f"خطأ أثناء التفريغ: {e}"

        row = {
            "اسم الشركة": _company or "",
            "اسم الموظف": _employee or "",
            "الوظيفة": _job or "",
            "الخبرة": _exp or "",
            "الاختصاص": _spec or "",
            "المقابلة (النص المفرغ)": text or "",
        }
        rows.append(row)
        progress.progress(i / len(uploaded_files))

    status.success("اكتمل التفريغ ✅")

    # ---------------------- إنشاء DataFrame وملفات التحميل ----------------------
    df = pd.DataFrame(rows, columns=["اسم الشركة", "اسم الموظف", "الوظيفة", "الخبرة", "الاختصاص", "المقابلة (النص المفرغ)"])

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

    # CSV (اختياري)
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ تحميل CSV (UTF-8)",
        data=csv_data,
        file_name=f"تفريغ_المقابلات_{now}.csv",
        mime="text/csv"
    )

# ---------------------- تلميحات ----------------------
with st.expander("💡 ملاحظات هامة"):
    st.markdown(
        """
- يعمل التطبيق على Streamlit Cloud بدون الحاجة إلى تثبيت مكتبات فيديو/صوت معقّدة.
- إذا لم يعمل النموذج **gpt-4o-transcribe** على حسابك، جرّب **whisper-1** من الشريط الجانبي.
- لا تقم بحفظ مفتاحك داخل الكود أو GitHub. استخدم **Secrets** أو الحقل المؤقت داخل التطبيق.
        """
    )
