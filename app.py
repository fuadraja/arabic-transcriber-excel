import io
import mimetypes
import requests
import pandas as pd
import streamlit as st

# ---------- إعداد الصفحة ----------
st.set_page_config(page_title="تفريغ المقابلات الصوتية", page_icon="🎙️", layout="wide")
st.title("🎙️ تفريغ المقابلات الصوتية (عربي) → Excel")

# ---------- التحقق من المفتاح ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("⚠️ لم يتم العثور على مفتاح OpenAI في الأسرار. اذهب إلى Settings → Secrets وأضف OPENAI_API_KEY.")
    st.stop()

# ---------- دوال مساعدة ----------
TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL_NAME = "whisper-1"  # نموذج Whisper الرسمي

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def safe_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"

def transcribe_arabic(file_bytes: bytes, filename: str) -> dict:
    """
    يعيد dict فيها:
      - ok: bool
      - text: النص المفرغ (إن نجح)
      - error: رسالة الخطأ (إن فشل)
    """
    try:
        files = {
            "file": (filename, file_bytes, safe_mime(filename))
        }
        data = {
            "model": MODEL_NAME,
            "language": "ar"  # إجبار اللغة عربية لتحسين الدقة
        }
        resp = requests.post(TRANSCRIBE_URL, headers=HEADERS, files=files, data=data, timeout=120)
        if resp.status_code == 200:
            text = resp.json().get("text", "").strip()
            return {"ok": True, "text": text}
        else:
            # إظهار رسالة خطأ واضحة من واجهة OpenAI
            try:
                err = resp.json()
            except Exception:
                err = {"error": {"message": resp.text}}
            return {"ok": False, "error": f"Error code: {resp.status_code} - {err}"}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Network error: {e}"}


# ---------- واجهة الإدخال العامة ----------
st.markdown("#### الخطوة 1: الإعدادات الافتراضية (يمكن تعديلها لكل ملف لاحقًا)")
colA, colB, colC, colD, colE = st.columns(5)
with colA:
    default_company = st.text_input("اسم الشركة (افتراضي)", value="")
with colB:
    default_employee = st.text_input("اسم الموظف (افتراضي)", value="")
with colC:
    default_role = st.text_input("الوظيفة (افتراضي)", value="")
with colD:
    default_experience = st.text_input("الخبرة (افتراضي)", value="")
with colE:
    default_specialty = st.text_input("الاختصاص (افتراضي)", value="")

st.markdown("#### الخطوة 2: ارفع الملفات الصوتية (يمكن رفع عدة ملفات)")
uploaded_files = st.file_uploader(
    "الملفات المدعومة: mp3, m4a, wav, webm, ogg …",
    type=["mp3", "m4a", "wav", "webm", "ogg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬆️ ارفع ملفًا واحدًا أو أكثر للمتابعة.")
    st.stop()

st.divider()

# سننشئ عناصر إدخال لكل ملف داخل Expander
st.markdown("#### الخطوة 3: أدخل بيانات كل ملف على حدة (اختياري)")
per_file_inputs = []
for idx, uf in enumerate(uploaded_files, start=1):
    with st.expander(f"🗂️ ملف {idx}: {uf.name}", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            company = st.text_input("اسم الشركة", value=default_company, key=f"company_{idx}")
        with c2:
            employee = st.text_input("اسم الموظف", value=default_employee, key=f"employee_{idx}")
        with c3:
            role = st.text_input("الوظيفة", value=default_role, key=f"role_{idx}")

        c4, c5 = st.columns(2)
        with c4:
            experience = st.text_input("الخبرة", value=default_experience, key=f"experience_{idx}")
        with c5:
            specialty = st.text_input("الاختصاص", value=default_specialty, key=f"specialty_{idx}")

        per_file_inputs.append({
            "filename": uf.name,
            "company": company.strip(),
            "employee": employee.strip(),
            "role": role.strip(),
            "experience": experience.strip(),
            "specialty": specialty.strip(),
            "file_obj": uf
        })

st.divider()

# ---------- تنفيذ التفريغ ----------
if st.button("🚀 بدء التفريغ لجميع الملفات", type="primary"):
    results = []
    errors = []

    progress = st.progress(0)
    status_area = st.empty()

    for i, item in enumerate(per_file_inputs, start=1):
        status_area.info(f"جاري تفريغ: {item['filename']} ({i}/{len(per_file_inputs)}) ...")
        file_bytes = item["file_obj"].read()
        item["file_obj"].seek(0)

        out = transcribe_arabic(file_bytes, item["filename"])
        if out["ok"]:
            row = {
                "اسم الملف": item["filename"],  # مفيد للتمييز بين الصفوف
                "اسم الشركة": item["company"],
                "اسم الموظف": item["employee"],
                "الوظيفة": item["role"],
                "الخبرة": item["experience"],
                "الاختصاص": item["specialty"],
                "المقابلة (النص المفرغ)": out["text"]
            }
            results.append(row)
        else:
            # نسجّل صفًا مع رسالة الخطأ (لو أردت إهمال الصف، احذف هذا القسم)
            row = {
                "اسم الملف": item["filename"],
                "اسم الشركة": item["company"],
                "اسم الموظف": item["employee"],
                "الوظيفة": item["role"],
                "الخبرة": item["experience"],
                "الاختصاص": item["specialty"],
                "المقابلة (النص المفرغ)": f"خطأ أثناء التفريغ: {out['error']}"
            }
            results.append(row)
            errors.append(f"❌ {item['filename']}: {out['error']}")

        progress.progress(i / len(per_file_inputs))

    status_area.empty()

    # ---------- عرض النتائج وجدول ملخّص ----------
    df = pd.DataFrame(results, columns=[
        "اسم الملف",
        "اسم الشركة",
        "اسم الموظف",
        "الوظيفة",
        "الخبرة",
        "الاختصاص",
        "المقابلة (النص المفرغ)"
    ])

    st.success(f"تمت معالجة {len(per_file_inputs)} ملف(ات).")
    if errors:
        with st.expander("عرض الأخطاء", expanded=False):
            for e in errors:
                st.error(e)

    st.subheader("📄 النتائج")
    st.dataframe(df, use_container_width=True, height=400)

    # ---------- إنشاء ملف Excel واحد للتنزيل ----------
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="تفريغ المقابلات", index=False)
    buffer.seek(0)

    st.download_button(
        label="⬇️ تنزيل Excel مجمّع",
        data=buffer,
        file_name="تفريغ_المقابلات.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("يحتوي ملف Excel على صف لكل ملف صوتي. ستجد اسم الملف في العمود الأول للتمييز.")


# ---------- ملاحظات ----------
with st.expander("ملاحظات وتسعير", expanded=False):
    st.markdown("""
- تأكد من أن **مفتاح OpenAI صحيح** ولديك **رصيد كافٍ**؛ أخطاء مثل:
  - `invalid_api_key` تعني أن المفتاح غير صحيح.
  - `insufficient_quota (429)` تعني لا يوجد رصيد كافٍ في الحساب.
- يدعم التطبيق ملفات `mp3, m4a, wav, webm, ogg`.
- الأعمدة النهائية في Excel:
  **اسم الشركة | اسم الموظف | الوظيفة | الخبرة | الاختصاص | المقابلة (النص المفرغ)**  
  تمت إضافة عمود **اسم الملف** لمساعدتك على التمييز بين الصفوف، ويمكنك حذفه لاحقًا إن رغبت.
""")
