# ========================= سحابة الكلمات (اختياري) =========================
with st.expander("☁️ سحابة الكلمات (Word Cloud)"):
    enable_wc = st.checkbox("تفعيل توليد سحابة الكلمات", value=True)
    col_wc1, col_wc2 = st.columns([2, 1])
    with col_wc1:
        st.caption("لأفضل عرض عربي، يُفضّل رفع خط عربي (TTF/OTF) مثل Noto Naskh أو Amiri.")
        font_file = st.file_uploader("📎 خط عربي (اختياري)", type=["ttf", "otf"], key="wc_font")
    with col_wc2:
        max_words = st.slider("عدد الكلمات القصوى", 50, 500, 200, 25)
        bg_white = st.checkbox("خلفية بيضاء", value=True)
    
    # قائمة توقفات عربية بسيطة (يمكن تعديلها)
    default_stopwords = """
    في على من إلى عن أن إن كان تكون كانوا تكونون هذا هذه ذلك تلك هناك هنا ثم لقد لقد قد مع أو ولا ولم لما ما لا ليس إنّه أنها إنه أنهم التي الذي الذين التي حيث بسبب جدًا جدا خلال بين حتى لدى لدى لدى لدى دون عند قبل بعد مثل أيضًا ايضا إذ اذ إذا اذا إذًا اذاً فقط كل أي اي كيف ماذا لماذا متى حيثما حينما حين أنّ انّ ألا الا ألّا ألّا و أو أم اما لكن بل سوى غير ضد ذات ضمن نحو عبر عبرًا ربما قد قدًا مزيد أقل أكثر جدا جداً جدًا
    """.split()
    user_stop = st.text_area("كلمات إيقاف إضافية (افصل بينها بمسافة)", value="", help="أضف كلمات لتجاهلها في السحابة.")
    # دمج الإيقاف
    extra_stop = [w.strip() for w in user_stop.split() if w.strip()]
    arabic_stopwords = set([w for w in default_stopwords + extra_stop if w])

    if enable_wc and not df.empty:
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            import arabic_reshaper
            from bidi.algorithm import get_display

            def build_wc_image(text: str, font_bytes: bytes | None) -> bytes:
                # تهيئة العربية (ربط الحروف + اتجاه)
                reshaped = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped)

                # استبعاد كلمات الإيقاف يدويًا (wordcloud لا يدعم الإعراب العربي بالكامل)
                tokens = [t for t in bidi_text.split() if t not in arabic_stopwords]
                cleaned_text = " ".join(tokens)

                # إعدادات الخط
                font_path = None
                tmp_font_path = None
                if font_bytes:
                    import tempfile
                    _, ext = os.path.splitext(font_file.name)
                    ext = ext or ".ttf"
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tf.write(font_bytes)
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

                # تنظيف ملف الخط المؤقت
                if tmp_font_path and os.path.exists(tmp_font_path):
                    try:
                        os.remove(tmp_font_path)
                    except Exception:
                        pass

                return buf.getvalue()

            st.subheader("سحابة الكلمات المجمّعة (كل التفريغات)")
            all_text = " ".join(df["المقابلة (النص المفرغ)"].astype(str).tolist())
            if all_text.strip():
                img_bytes = build_wc_image(all_text, font_file.read() if font_file else None)
                st.image(img_bytes, use_container_width=True)
                st.download_button(
                    "⬇️ تحميل السحابة المجمّعة (PNG)",
                    data=img_bytes,
                    file_name=f"wordcloud_all_{now}.png",
                    mime="image/png"
                )
            else:
                st.info("لا يوجد نص لتوليد سحابة مجمّعة.")

            st.markdown("---")
            st.subheader("سحابة كلمات لكل ملف")
            tabs = st.tabs([f"ملف {i+1}" for i in range(len(df))])
            for i, tab in enumerate(tabs):
                with tab:
                    text_i = str(df.iloc[i]["المقابلة (النص المفرغ)"])
                    meta_label = " | ".join(str(df.iloc[i][c]) for c in ["اسم الشركة", "اسم الموظف", "الوظيفة"] if c in df.columns)
                    st.caption(meta_label)
                    if text_i.strip():
                        img_i = build_wc_image(text_i, font_file.read() if font_file else None)
                        st.image(img_i, use_container_width=True)
                        st.download_button(
                            f"⬇️ تحميل سحابة ملف #{i+1} (PNG)",
                            data=img_i,
                            file_name=f"wordcloud_file_{i+1}_{now}.png",
                            mime="image/png",
                            key=f"dl_wc_{i}"
                        )
                    else:
                        st.info("لا يوجد نص في هذا الصف.")
        except Exception as e:
            st.warning(f"تعذر توليد سحابة الكلمات: {e}")
