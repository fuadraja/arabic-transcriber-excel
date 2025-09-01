# app.py
import os
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd

# ---------------------- إعدادات الصفحة ----------------------
st.set_page_config(page_title="تفريغ المقابلات إلى إكسل", page_icon="📝", layout="wide")
st.title("📝 تفريغ المقابلات الصوتية (عربي) → إكسل")
st.caption("ارفع تسجيلات بالعربية لنُفرِّغها نصيًا ونحفظ النتائج في إكسل بالأعمدة المطلوبة. يدعم رفع عدة ملفات وإدخال بيانات مختلفة لكل ملف.")

# ---------------------- مفتاح OpenAI ----------------------
from openai import OpenAI

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

# ---------------------- إعدادات في الشريط الجانبي ----------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    model = st.selectbox(
        "نموذج التفريغ (يفضّل الأول)",
        options=["gpt-4o-transcribe", "whisper-1"],
        index=0,
        help="اختر gpt-4o-transcribe إن كان متاحًا في حسابك؛ وإلا استخدم whisper-1."
    )
    temperature = st.slider("Temperature (اختياري)", 0.0, 1.0, 0.0, 0.1)
    st.caption("اتركها 0 للحصول على نص أدقّ بدون تنويعات.")
    st.markdown("---")
    st.caption("📎 الصيغ المدعومة: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm")

# ---------------------- نموذج إدخال البيانات + رفع الملفات ----------------------
with st.form("meta_form", clear_on_submit=False):
    st.subheader("🧾 بيانات افتراضية (يمكن تطبيقها على جميع الملفات)")
    col1, col2, col3 = st.columns(3)
    with col1:
        default_company = st.text_input("اسم الشركة (افتراضي)")
        default_employee = st.text_input("اسم الموظف (افتراضي)")
    with col2:
        default_job_title = st.text_input("الوظيفة (افتراضي)")
        default_experience = st.text_input("الخبرة (افتراضي)", placeholder="مثال: 5 سنوات")
    with col3:
        default_specialization = st.text_input("الاختصاص (افتراضي)")
        auto_fill = st.checkbox("تطبيق القيم الافتراضية على جميع الملفات", value=True)

    uploaded_files = st.file_uploader(
        "🎧 ارفع ملفات الصوت (يمكن اختيار أكثر من ملف)",
        type=["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"],
        accept_multiple_files=True
    )

    # عند عدم استخدام التعبئة التلقائية، نظهر حقولًا لكل ملف
    if uploaded_files and not auto_fill:
        st.markdown("### ✏️ بيانات لكل ملف")
        for idx, f in enumerate(uploaded_files, start=1):
            with st.expander(f"الملف #{idx} — {f.name}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.text_input("اسم الشركة", key=f"company_{idx}", value=default_company)
                    st.text_input("اسم الموظف", key=f"employee_{idx}", value=default_employee)
                with c2:
                    st.text_input("الوظيفة", key=f"job_{idx}", value=default_job_title)
                    st.text_input("الخبرة", key=f"exp_{idx}", value=default_experience)
                with c3:
                    st.text_input("الاختصاص", key=f"spec_{idx}", value=default_specialization)

    submit = st.form_submit_button("بدء التفريغ ▶️")

# ---------------------- الدالة: تفريغ ملف ----------------------
def transcribe_file(client: OpenAI, file_obj, model_name: str, temperature_value: float = 0.0) -> str:
    """
    يُعيد نصًا مُفرّغًا باللغة العربية من ملف صوتي، باستخدام واجهة OpenAI الحديثة.
    """
    # نقرأ المحتوى مرة واحدة ثم نعيد المؤشر
    content = file_obj.read()
    file_obj.seek(0)
    try:
        resp = client.audio.transcriptions.create(
            model=model_name,
            file=(file_obj.name, content),
            language="ar",
            temperature=temperature_value,
            response_format="text",
        )
        if isinstance(resp, str):
            return resp.strip()
        return getattr(resp, "text", "").strip()
    except Exception as e:
        # تُعاد الرسالة كـ نصّ ضمن الحقل
        return f"خطأ أثناء التفريغ: {e}"

# ---------------------- التنفيذ ----------------------
df = None
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

    total = len(uploaded_files)
    for i, f in enumerate(uploaded_files, start=1):
        status.info(f"جارٍ تفريغ: {f.name} ...")

        # اجلب القيم الخاصة بكل ملف إن كانت متاحة، وإلا استخدم الافتراضية
        if auto_fill:
            _company = default_company
            _employee = default_employee
            _job = default_job_title
            _exp = default_experience
            _spec = default_specialization
        else:
            _company = st.session_state.get(f"company_{i}", default_company)
            _employee = st.session_state.get(f"employee_{i}", default_employee)
            _job = st.session_state.get(f"job_{i}", default_job_title)
            _exp = st.session_state.get(f"exp_{i}", default_experience)
            _spec = st.session_state.get(f"spec_{i}", default_specialization)

        # التفريغ
        text = transcribe_file(client, f, model, temperature)

        row = {
            "اسم الشركة": _company or "",
            "اسم الموظف": _employee or "",
            "الوظيفة": _job or "",
            "الخبرة": _exp or "",
            "الاختصاص": _spec or "",
            "المقابلة (النص المفرغ)": text or "",
        }
        rows.append(row)
        progress.progress(i / total)

    status.success("اكتمل التفريغ ✅")

    # -------- إنشاء DataFrame وملفات التحميل --------
    df = pd.DataFrame(
        rows,
        columns=["اسم الشركة", "اسم الموظف", "الوظيفة", "الخبرة", "الاختصاص", "المقابلة (النص المفرغ)"]
    )

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

# ========================= سحابة الكلمات (Word Cloud) =========================
if df is not None and not df.empty:
    with st.expander("☁️ سحابة الكلمات (Word Cloud)"):
        enable_wc = st.checkbox("تفعيل توليد سحابة الكلمات", value=True)
        col_wc1, col_wc2 = st.columns([2, 1])
        with col_wc1:
            st.caption("لأفضل عرض عربي، يُفضّل رفع خط عربي (TTF/OTF) مثل Noto Naskh أو Amiri.")
            font_file = st.file_uploader("📎 خط عربي (اختياري)", type=["ttf", "otf"], key="wc_font")
        with col_wc2:
            max_words = st.slider("عدد الكلمات القصوى", 50, 500, 200, 25)
            bg_white = st.checkbox("خلفية بيضاء", value=True)

        # كلمات إيقاف عربية بسيطة + ما يضيفه المستخدم
        default_stopwords = """
        في على من إلى عن أن إن كان تكون كانوا تكونون هذا هذه ذلك تلك هناك هنا ثم لقد قد مع أو ولا ولم لما ما لا ليس إنه أنها إنهم التي الذي الذين حيث بسبب جدا جدًا خلال بين حتى لدى دون عند قبل بعد مثل أيضًا ايضا إذ اذا إذًا فقط كل أي اي كيف ماذا لماذا متى حينما حيثما أنَّ إنَّ
        """.split()
        user_stop = st.text_area("كلمات إيقاف إضافية (افصل بينها بمسافة)", value="")
        extra_stop = [w.strip() for w in user_stop.split() if w.strip()]
        arabic_stopwords = set([w for w in default_stopwords + extra_stop if w])

        if enable_wc:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                import arabic_reshaper
                from bidi.algorithm import get_display

                font_bytes = font_file.read() if font_file is not None else None

                def build_wc_image(text: str, font_bytes_local: bytes | None) -> bytes:
                    # تهيئة العربية (ربط الحروف + اتجاه)
                    reshaped = arabic_reshaper.reshape(text)
                    bidi_text = get_display(reshaped)

                    # استبعاد كلمات الإيقاف يدويًا
                    tokens = [t for t in bidi_text.split() if t not in arabic_stopwords]
                    cleaned_text = " ".join(tokens)

                    # إعداد الخط (حفظ مؤقت إذا رُفع)
                    font_path = None
                    tmp_font_path = None
                    if font_bytes_local:
                        import tempfile
                        suffix = ".ttf"
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tf.write(font_bytes_local)
                        tf.flush()
                        tf.close()
                        tmp_font_path = tf.name
                        font_path = tmp_font_path

                    # توليد السحابة
                    wc = WordCloud(
                        width=1200,
                        height=600,
                        background_color="white" if bg_white else None,
                        mode="RGBA" if not bg_white else "RGB",
                        max_words=int(max_words),
                        font_path=font_path,
                        collocations=False,
                    ).generate(cleaned_text)

                    # رسم وحفظ PNG
                    buf = BytesIO()
                    fig = plt.figure(figsize=(10, 5))
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                    buf.seek(0)

                    if tmp_font_path and os.path.exists(tmp_font_path):
                        try:
                            os.remove(tmp_font_path)
                        except Exception:
                            pass

                    return buf.getvalue()

                # سحابة مجمّعة
                st.subheader("سحابة الكلمات المجمّعة (كل التفريغات)")
                all_text = " ".join(df["المقابلة (النص المفرغ)"].astype(str).tolist())
                if all_text.strip():
                    img_bytes = build_wc_image(all_text, font_bytes)
                    st.image(img_bytes, use_container_width=True)
                    st.download_button(
                        "⬇️ تحميل السحابة المجمّعة (PNG)",
                        data=img_bytes,
                        file_name=f"wordcloud_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                else:
                    st.info("لا يوجد نص لتوليد سحابة مجمّعة.")

                st.markdown("---")

                # سحب سحابة لكل صف/ملف
                st.subheader("سحابة كلمات لكل ملف")
                tabs = st.tabs([f"ملف {i+1}" for i in range(len(df))])
                for i, tab in enumerate(tabs):
                    with tab:
                        text_i = str(df.iloc[i]["المقابلة (النص المفرغ)"])
                        meta_label = " | ".join(
                            str(df.iloc[i][c]) for c in ["اسم الشركة", "اسم الموظف", "الوظيفة"]
                            if c in df.columns
                        )
                        st.caption(meta_label)
                        if text_i.strip():
                            img_i = build_wc_image(text_i, font_bytes)
                            st.image(img_i, use_container_width=True)
                            st.download_button(
                                f"⬇️ تحميل سحابة ملف #{i+1} (PNG)",
                                data=img_i,
                                file_name=f"wordcloud_file_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                key=f"dl_wc_{i}"
                            )
                        else:
                            st.info("لا يوجد نص في هذا الصف.")
            except Exception as e:
                st.warning(f"تعذر توليد سحابة الكلمات: {e}")
else:
    st.info("لا توجد بيانات نصية بعد لتوليد سحابة الكلمات.")

# ---------------------- تلميحات ----------------------
with st.expander("💡 ملاحظات هامة"):
    st.markdown(
        """
- يعمل التطبيق على Streamlit Cloud بدون الحاجة إلى مكتبات صوت/فيديو إضافية.
- إذا لم يعمل النموذج **gpt-4o-transcribe** على حسابك، جرّب **whisper-1** من الشريط الجانبي.
- لا تحفظ مفتاحك داخل الكود أو GitHub. استخدم **Secrets** أو الحقل المؤقت داخل التطبيق.
        """
    )
