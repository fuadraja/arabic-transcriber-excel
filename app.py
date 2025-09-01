import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import math
import json
from openai import OpenAI

# ====================== إعدادات الصفحة ======================
st.set_page_config(page_title="تفريغ + تحليل المقابلات", page_icon="📝", layout="wide")
st.title("📝 تفريغ المقابلات الصوتية (عربي) + تحليل نصي تلقائي → إكسل")

st.caption("ارفع عدة ملفات صوتية بالعربية، أدخل بيانات كل ملف، سنفرّغ النص ونحلّله تلقائيًا ونصدر النتائج إلى ملف إكسل واحد.")

# ====================== مفتاح OpenAI ======================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
if not OPENAI_API_KEY:
    with st.expander("🔐 أدخل OpenAI API Key (مؤقت لهذه الجلسة إن لم تستطع استخدام Secrets)"):
        OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", type="password", placeholder="sk-********************************")
if not OPENAI_API_KEY:
    st.info("يفضل إضافة المفتاح في Settings → Secrets على Streamlit Cloud.", icon="🔑")

# ====================== الشريط الجانبي: إعدادات + حاسبة تكلفة ======================
with st.sidebar:
    st.header("⚙️ الإعدادات")

    trx_model = st.selectbox(
        "نموذج التفريغ الصوتي",
        options=["whisper-1", "gpt-4o-transcribe"],  # جرّب whisper-1 أولاً
        index=0,
        help="إن لم يعمل gpt-4o-transcribe على حسابك استخدم whisper-1."
    )
    nlp_model = st.selectbox(
        "نموذج التحليل النصي",
        options=["gpt-4o-mini", "gpt-4o-mini-translate"],  # كلاهما اقتصادي
        index=0,
        help="نموذج اقتصادي لاستخراج الملخص/المهارات/المشاعر."
    )
    temperature = st.slider("Temperature (للتحليل فقط)", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.subheader("💰 حاسبة التكلفة التقديرية")

    # أسعار تقريبية (كما ناقشنا)
    WHISPER_PER_MIN = 0.006  # $/minute
    GPT_INPUT_PER_1K = 0.00015
    GPT_OUTPUT_PER_1K = 0.00060

    calc_files = st.number_input("عدد الملفات", min_value=1, value=10, step=1)
    calc_avg_minutes = st.number_input("متوسط الدقائق لكل ملف", min_value=0.0, value=10.0, step=1.0)
    calc_avg_tokens_in = st.number_input("متوسط Tokens للنص/ملف (اختياري)", min_value=0, value=2000, step=500,
                                         help="إن تركته كما هو سنستخدم 2K كمتوسط.")
    calc_avg_tokens_out = st.number_input("متوسط Tokens للمخرجات/ملف (اختياري)", min_value=0, value=500, step=100)

    est_trx_cost = calc_files * calc_avg_minutes * WHISPER_PER_MIN
    est_nlp_cost = calc_files * ((calc_avg_tokens_in / 1000) * GPT_INPUT_PER_1K + (calc_avg_tokens_out / 1000) * GPT_OUTPUT_PER_1K)
    st.metric("تكلفة التفريغ (تقديري)", f"${est_trx_cost:,.2f}")
    st.metric("تكلفة التحليل (تقديري)", f"${est_nlp_cost:,.2f}")
    st.metric("الإجمالي (تقديري)", f"${(est_trx_cost + est_nlp_cost):,.2f}")

# ====================== رفع الملفات + جدول بيانات لكل ملف ======================
st.subheader("📤 رفع الملفات وإدخال بيانات كل ملف")
uploaded_files = st.file_uploader(
    "ارفع ملفات الصوت (MP3/WAV/M4A/MP4). يمكن اختيار أكثر من ملف",
    type=["mp3", "wav", "m4a", "mp4"],
    accept_multiple_files=True
)

# نبني جدول بيانات قابل للتعديل لإدخال البيانات لكل ملف
file_rows = []
if uploaded_files:
    for idx, f in enumerate(uploaded_files, start=1):
        file_rows.append({
            "index": idx,
            "اسم الملف": f.name,
            "اسم الشركة": "",
            "اسم الموظف": "",
            "الوظيفة": "",
            "الخبرة": "",
            "الاختصاص": "",
            "المدة (دقيقة) - اختياري": 0.0  # لأجل حساب التكلفة/الملخص لاحقًا لو أردت
        })

    df_meta = pd.DataFrame(file_rows)
    st.info("عدّل القيم في الجدول أدناه لكل ملف قبل البدء.", icon="✏️")
    df_meta = st.data_editor(
        df_meta,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )
else:
    df_meta = pd.DataFrame(columns=["index","اسم الملف","اسم الشركة","اسم الموظف","الوظيفة","الخبرة","الاختصاص","المدة (دقيقة) - اختياري"])

# ====================== دوال OpenAI ======================
def transcribe_file(client: OpenAI, file_obj, model_name: str) -> str:
    """
    يعيد نصًا عربيًا مُفرّغًا من ملف صوتي باستخدام OpenAI.
    """
    # نقرأ البايتات للحفاظ على الملف للاستخدامات الأخرى إن لزم
    audio_bytes = file_obj.read()
    file_obj.seek(0)

    # واجهة التفريغ
    result = client.audio.transcriptions.create(
        model=model_name,
        file=(file_obj.name, audio_bytes),
        language="ar",
        response_format="text",
        temperature=0.0
    )
    # result نص خام عند response_format="text"
    return result.strip() if isinstance(result, str) else getattr(result, "text", "").strip()

def approx_tokens_from_text(text: str) -> int:
    # تقدير سريع: 1 token ≈ 4 أحرف عربية/لاتينية تقريبًا
    return max(1, math.ceil(len(text) / 4))

def analyze_text(client: OpenAI, text: str, model_name: str, temperature: float = 0.0) -> dict:
    """
    يولّد مخرجات تحليلية منظّمة JSON: ملخص، مشاعر، مهارات، ملاحظات/أعلام حمراء.
    نستخدم JSON mode لضمان هيكل ثابت.
    """
    sys = (
        "أنت محلل نصوص عربي. استخرج ملخصًا موجزًا، المشاعر العامة (موجب/محايد/سالب مع درجة 0-1)، "
        "قائمة مهارات مستنتجة، وأي ملاحظات/أعلام حمراء مهمّة. أعد النتيجة كـ JSON فقط."
    )
    user = f"""
النص العربي التالي هو تفريغ لمقابلة:
{text}
ارجع JSON بالمخطط:
{{
  "summary": "ملخص موجز بالعربية (3-6 أسطر)",
  "sentiment": {{"label": "موجب|محايد|سالب", "score": 0.0}},
  "skills": ["مهارة1","مهارة2", "..."],
  "red_flags": ["ملاحظة", "..."]
}}
    """.strip()

    resp = client.responses.create(
        model=model_name,
        temperature=temperature,
        reasoning={"effort": "low"},
        response_format={"type": "json_object"},
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    )

    # استخراج النص من المخرجات
    content = ""
    if resp.output and len(resp.output) > 0 and hasattr(resp.output[0], "content") and len(resp.output[0].content) > 0:
        content = resp.output[0].content[0].text or ""
    else:
        # احتياط لواجهات مختلفة
        content = getattr(resp, "output_text", "") or ""

    try:
        data = json.loads(content) if content else {}
    except Exception:
        data = {"summary": "", "sentiment": {"label": "محايد", "score": 0.0}, "skills": [], "red_flags": []}
    return data

# ====================== أزرار التنفيذ ======================
col_run1, col_run2 = st.columns([1,1])
with col_run1:
    do_transcribe = st.button("▶️ بدء التفريغ + التحليل")
with col_run2:
    st.write("")

if do_transcribe:
    if not OPENAI_API_KEY:
        st.error("الرجاء إدخال مفتاح OpenAI API أولًا.", icon="🚫")
        st.stop()
    if uploaded_files is None or len(uploaded_files) == 0:
        st.warning("الرجاء رفع الملفات أولًا.", icon="📎")
        st.stop()

    client = OpenAI(api_key=OPENAI_API_KEY)

    # خرْج نهائي
    rows_result = []
    rows_analysis = []

    # تقدّم
    progress = st.progress(0)
    status = st.empty()

    # نبني خريطة من اسم الملف → بياناته من الجدول
    meta_map = {r["اسم الملف"]: r for _, r in df_meta.iterrows()} if not df_meta.empty else {}

    # تكاليف فعلية تقريبية (نقدّر tokens من طول النص)
    total_minutes = 0.0
    total_trx_cost = 0.0
    total_nlp_cost = 0.0

    for i, f in enumerate(uploaded_files, start=1):
        status.info(f"جارٍ معالجة: {f.name} …")

        # بيانات هذا الملف من الجدول
        meta = meta_map.get(f.name, {})
        company = meta.get("اسم الشركة", "")
        employee = meta.get("اسم الموظف", "")
        job_title = meta.get("الوظيفة", "")
        experience = meta.get("الخبرة", "")
        specialization = meta.get("الاختصاص", "")
        minutes_opt = float(meta.get("المدة (دقيقة) - اختياري", 0.0) or 0.0)

        # 1) التفريغ
        try:
            text = transcribe_file(client, f, trx_model)
        except Exception as e:
            text = f"خطأ أثناء التفريغ: {e}"

        # 2) التحليل
        analysis = {"summary": "", "sentiment": {"label": "", "score": 0.0}, "skills": [], "red_flags": []}
        if text and not text.startswith("خطأ أثناء التفريغ"):
            try:
                analysis = analyze_text(client, text, nlp_model, temperature=temperature)
            except Exception as e:
                analysis = {"summary": f"خطأ أثناء التحليل: {e}", "sentiment": {"label": "محايد", "score": 0.0}, "skills": [], "red_flags": []}

        # 3) تجميع صف النتائج العربية
        rows_result.append({
            "اسم الشركة": company,
            "اسم الموظف": employee,
            "الوظيفة": job_title,
            "الخبرة": experience,
            "الاختصاص": specialization,
            "المقابلة (النص المفرغ)": text
        })

        # 4) صف التحليل
        rows_analysis.append({
            "اسم الملف": f.name,
            "ملخص": analysis.get("summary", ""),
            "المشاعر - التصنيف": (analysis.get("sentiment") or {}).get("label", ""),
            "المشاعر - الدرجة": (analysis.get("sentiment") or {}).get("score", 0.0),
            "المهارات (قائمة)": ", ".join(analysis.get("skills") or []),
            "أعلام/ملاحظات": ", ".join(analysis.get("red_flags") or [])
        })

        # 5) تقدير التكلفة الفعلية (تقريبية)
        # التفريغ: إن أدخل المستخدم مدة الملف نستخدمها، وإلا لا نحسب تفصيليًا (لأننا لا نستخرج المدة دون حزم صوت)
        if minutes_opt > 0:
            total_minutes += minutes_opt
        # تحليل: نقدّر tokens من طول النص المفرغ
        t_in = approx_tokens_from_text(text) if text else 0
        # نفترض مخرجات التحليل ~ 500 tokens كمعدل إذا لم نستطع قياسها
        t_out = max(500, approx_tokens_from_text(json.dumps(analysis)))
        total_nlp_cost += (t_in / 1000) * GPT_INPUT_PER_1K + (t_out / 1000) * GPT_OUTPUT_PER_1K

        progress.progress(i / len(uploaded_files))

    # تكلفة التفريغ حسب الدقائق المُدخلة
    total_trx_cost = total_minutes * WHISPER_PER_MIN
    status.success("اكتملت المعالجة ✅")

    # ====================== عرض الجداول ======================
    df_results = pd.DataFrame(rows_result, columns=["اسم الشركة","اسم الموظف","الوظيفة","الخبرة","الاختصاص","المقابلة (النص المفرغ)"])
    df_analysis = pd.DataFrame(rows_analysis, columns=["اسم الملف","ملخص","المشاعر - التصنيف","المشاعر - الدرجة","المهارات (قائمة)","أعلام/ملاحظات"])

    st.subheader("📄 النتائج")
    st.dataframe(df_results, use_container_width=True)

    st.subheader("🧠 التحليلات")
    st.dataframe(df_analysis, use_container_width=True)

    # ====================== التصدير إلى إكسل ======================
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="النتائج")
        df_analysis.to_excel(writer, index=False, sheet_name="التحليلات")
        # ورقة ملخّص
        df_summary = pd.DataFrame([{
            "عدد الملفات": len(uploaded_files),
            "مجموع الدقائق (مُدخل)": total_minutes,
            "تكلفة التفريغ (تقديري $)": round(total_trx_cost, 4),
            "تكلفة التحليل (تقديري $)": round(total_nlp_cost, 4),
            "الإجمالي (تقديري $)": round(total_trx_cost + total_nlp_cost, 4),
            "نموج التفريغ": trx_model,
            "نموذج التحليل": nlp_model
        }])
        df_summary.to_excel(writer, index=False, sheet_name="الملخّص")
    excel_buffer.seek(0)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_name = f"تفريغ_وملفات_تحليل_{now}.xlsx"

    st.download_button(
        "⬇️ تحميل ملف إكسل شامل",
        data=excel_buffer,
        file_name=excel_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ====================== ملاحظات ======================
with st.expander("💡 ملاحظات"):
    st.markdown(
        """
- استخدم **whisper-1** للتفريغ لتفادي مشاكل التبعيات.  
- أدخل مدة كل ملف (اختياري) في الجدول لتحصل على **تكلفة تفريغ أدق** في ورقة الملخّص.  
- التحليل يستخدم **JSON mode** لاستخراج ملخص/مشاعر/مهارات/ملاحظات بالعربية.  
- لا تحفظ المفتاح في الكود أو GitHub — استعمل **Secrets** في Streamlit Cloud.
        """
    )
