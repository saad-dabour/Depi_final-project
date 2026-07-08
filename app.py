"""
AI Skin Analyzer — Streamlit UI
================================
Run with:  streamlit run app.py
Requires:  best_model.h5 in the same directory +groq api.
"""

import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Skin Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL CODE  (copied verbatim from notebook — NOT modified)
# ─────────────────────────────────────────────────────────────────────────────

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
import numpy as np

@st.cache_resource(show_spinner="Loading AI model...")
def _load_model():

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(256, 256, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(5, activation="softmax")
    ])

    model.build((None, 256, 256, 3))
    model.load_weights("model/best_model.h5")

    return model


model = _load_model()

classes = [
    "Acne",         
    "Dark Spots",   
    "Wrinkles",     
    "Pores",        
    "Blackheads"   
]

def predict_skin_condition(pil_img):

    img = pil_img.convert("RGB")
    img = img.resize((256, 256))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return classes[idx], confidence

from groq import Groq

client = Groq(api_key="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")

def ask_llm(condition, confidence, user_question):

    prompt = f"""
You are a professional skincare assistant.

The AI detected:
- Skin condition: {condition}
- Confidence: {confidence*100:.2f}%

User question:
{user_question}

Answer the user's specific question naturally, like a dermatologist having a conversation.

Guidelines:
- Only include information that is relevant to the user's question.
- Don't repeat the diagnosis or confidence unless the user asks.
- Don't repeat a full skincare routine every time.
- If the user asks about products, recommend ingredients and products.
- If the user asks about recovery time, answer only about recovery time.
- If the user asks about causes, explain the causes.
- Suggest seeing a dermatologist only when appropriate (severe symptoms, worsening condition, uncertainty, or if asked about treatment).
- Keep the response conversational, concise, and easy to understand.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional skincare assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

# ── Page config ──────────────────────────────────────────────────────────────
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
html, body {
    color:#1f2937;
    font-family:"Poppins","Segoe UI",sans-serif;
}
[data-testid="stAppViewContainer"] { background: #faf9fc; }
[data-testid="stSidebar"] { background: #1e1b2e; }
[data-testid="stSidebar"] * { color: #e8e2f8 !important; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
    border-radius: 18px;
    padding: 2.2rem 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(124,58,237,.25);
}
.hero h1 { font-size: 2.4rem; margin: 0 0 .4rem; letter-spacing: -1px; }
.hero p  { font-size: 1.05rem; opacity: .88; margin: 0; }

/* ── upload box ── */
.upload-hint {
    border: 2px dashed #a855f7;
    border-radius: 14px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    background: #f5f0ff;
    color: #6b21a8;
    font-size: .95rem;
}

/* ── result card ── */
.result-card {
    background:white;
    color:#222222;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    border-left: 6px solid #7c3aed;
    box-shadow: 0 4px 20px rgba(0,0,0,.07);
    margin-bottom: 1rem;
}
.result-card h2 { margin: 0 0 .5rem; font-size: 1.7rem; }
.result-card p  { margin: .25rem 0; color: #444; }

/* ── severity badges ── */
.sev-mild     { color: #16a34a; font-weight: 700; font-size: 1.05rem; }
.sev-moderate { color: #d97706; font-weight: 700; font-size: 1.05rem; }
.sev-severe   { color: #dc2626; font-weight: 700; font-size: 1.05rem; }

/* ── recommendation cards ── */
.rec-card {
    background:#FFF8E1;
    color:#222222;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: .7rem;
    border: 1px solid #fde68a;
    font-size: .93rem;
    line-height: 1.55;
}
.rec-card-purple {
    background:#F3E8FF;
    color:#222222;
    border: 1px solid #d8b4fe;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: .7rem;
    font-size: .93rem;
    line-height: 1.55;
}
.rec-card-pink {
    background:#FCE7F3;
    color:#222222;
    border: 1px solid #fbb6d3;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: .7rem;
    font-size: .93rem;
    line-height: 1.55;
}

/* ── section title ── */
.section-title {
    font-size:28px;
    font-weight:700;
    color:#4C1D95;
    border-bottom: 2px solid #ede9fe;
    padding-bottom: .4rem;
    margin-bottom: 1rem;
}

/* ── chat bubbles ── */
.chat-user {
    background: #7c3aed;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: .75rem 1.1rem;
    margin: .5rem 0 .5rem 20%;
    font-size: .95rem;
    line-height: 1.5;
}
.chat-bot {
    background: white;
    color: #1e1b2e;
    border: 1px solid #e5e7eb;
    border-radius: 18px 18px 18px 4px;
    padding: .75rem 1.1rem;
    margin: .5rem 20% .5rem 0;
    font-size: .95rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.chat-label { font-size: .78rem; color: #9ca3af; margin-bottom: .15rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 AI Skin Analyzer</h1>
    <p>Upload a skin image · Get an instant AI diagnosis · Receive personalized skincare advice</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧴 AI Skin Analyzer")
    st.markdown("---")
    st.markdown("""
**Detectable Conditions**

| Icon | Condition |
|------|-----------|
| 🔴 | Acne |
| ⚫ | Blackheads |
| 🟤 | Dark Spots |
| 🔵 | Large Pores |
| 〰️ | Wrinkles |
""")
    st.markdown("---")
    st.markdown("""
**How to use**
1. Upload a clear skin photo
2. Click **Analyze Skin**
3. View your AI diagnosis
4. Read personalized recommendations
5. Chat with the AI assistant

---
⚠️ *For informational purposes only. Always consult a certified dermatologist for medical diagnosis and treatment.*
""")
    st.markdown("---")
    st.markdown("**Model:** VGG16 · Transfer Learning  \n**Accuracy:** ~95%  \n**Chatbot:** Llama 3.3 via Groq")

# ── Skincare recommendation data ──────────────────────────────────────────────
SKINCARE_INFO = {
    "Acne": {
        "icon": "🔴",
        "description": (
            "Acne occurs when hair follicles become clogged with oil and dead skin cells, "
            "resulting in pimples, blackheads, and whiteheads. It is most common on the face, "
            "forehead, chest, and shoulders."
        ),
        "routine": [
            "🌅 AM: Gentle foaming cleanser → Alcohol-free toner → Lightweight moisturizer → SPF 30+",
            "🌙 PM: Double cleanse → Salicylic acid serum → Spot treatment (benzoyl peroxide) → Oil-free moisturizer",
        ],
        "ingredients": ["Salicylic Acid (BHA)", "Benzoyl Peroxide", "Niacinamide", "Tea Tree Oil", "Retinol (PM only)"],
        "avoid": ["Heavy comedogenic oils", "Harsh physical scrubs", "Over-washing (strips barrier)", "Touching your face"],
        "tips": [
            "💤 Change pillowcases at least twice a week",
            "🚫 Never pop pimples — it causes scarring and spreads bacteria",
            "💧 Drink plenty of water and maintain a balanced diet",
            "🧴 Choose 'non-comedogenic' labeled products",
        ],
    },
    "Blackheads": {
        "icon": "⚫",
        "description": (
            "Blackheads are open comedones — pores clogged with excess sebum and dead skin cells "
            "that oxidize and turn dark when exposed to air. They commonly appear on the nose, chin, and forehead."
        ),
        "routine": [
            "🌅 AM: Oil-free gel cleanser → BHA toner → Non-comedogenic moisturizer → SPF 30+",
            "🌙 PM: Oil cleanser (dissolves sebum plugs) → BHA exfoliant 2–3×/week → Clay mask 1×/week → Light moisturizer",
        ],
        "ingredients": ["Salicylic Acid (BHA)", "Niacinamide", "Retinol", "Witch Hazel", "Kaolin / Bentonite Clay"],
        "avoid": ["Pore strips (cause micro-tears)", "Harsh abrasive scrubs", "Heavy creams and mineral oil"],
        "tips": [
            "♨️ Steam your face for 5 min to loosen plugs before cleansing",
            "🧖 Use a BHA exfoliant 2–3 times a week — not daily",
            "👆 Never squeeze blackheads manually",
            "🛁 Double cleanse in the evening to fully remove oil buildup",
        ],
    },
    "Dark Spots": {
        "icon": "🟤",
        "description": (
            "Dark spots (hyperpigmentation) are areas where excess melanin has accumulated, "
            "leaving patches darker than the surrounding skin. Causes include sun damage, post-acne marks, "
            "hormonal changes, and aging."
        ),
        "routine": [
            "🌅 AM: Gentle cleanser → Vitamin C serum → Moisturizer → SPF 50+ (non-negotiable!)",
            "🌙 PM: Cleanser → Alpha Arbutin or Niacinamide serum → Retinol 2–3×/week → Rich moisturizer",
        ],
        "ingredients": ["Vitamin C (L-Ascorbic Acid)", "Niacinamide", "Alpha Arbutin", "Kojic Acid", "Tranexamic Acid", "AHA (Glycolic / Lactic Acid)"],
        "avoid": ["Skipping SPF (UV darkens spots further)", "Picking at skin", "Prolonged unprotected sun exposure"],
        "tips": [
            "☀️ SPF is your single most effective dark-spot treatment",
            "⏳ Be consistent — fading spots takes 8–12 weeks minimum",
            "🧢 Wear a hat outdoors for extra UV protection",
            "🍊 Vitamin C in the morning + SPF = powerful brightening combo",
        ],
    },
    "Large Pores": {
        "icon": "🔵",
        "description": (
            "Large pores are expanded hair follicle openings that become more visible when clogged "
            "with sebum, dead skin, or bacteria. Genetics, oily skin, and aging are the main contributors."
        ),
        "routine": [
            "🌅 AM: Foam / gel cleanser → Niacinamide serum → Light gel moisturizer → SPF 30+",
            "🌙 PM: Oil cleanser → AHA/BHA toner 3×/week → Retinol (build up gradually) → Moisturizer",
        ],
        "ingredients": ["Niacinamide", "Retinol", "Salicylic Acid (BHA)", "Glycolic Acid (AHA)", "Kaolin / Bentonite Clay"],
        "avoid": ["Heavy oils and occlusive butters", "Comedogenic products", "Skipping evening cleanse"],
        "tips": [
            "🕯️ Retinol minimizes pores over time — be patient, it takes months",
            "🧊 Finish with a cold-water rinse after cleansing to tighten pores",
            "💄 Use a pore-filling primer if wearing makeup",
            "🛁 Keep skin consistently clean to prevent pores from stretching",
        ],
    },
    "Wrinkles": {
        "icon": "〰️",
        "description": (
            "Wrinkles are creases and folds in the skin caused by a natural decline in collagen and elastin "
            "with age, amplified by UV exposure, smoking, dehydration, and repetitive facial movements."
        ),
        "routine": [
            "🌅 AM: Gentle cleanser → Antioxidant serum (Vitamin C/E) → Hyaluronic Acid → Peptide moisturizer → SPF 50+",
            "🌙 PM: Double cleanse → Retinol (start at 0.025%, build slowly) → Rich barrier moisturizer → Neck cream",
        ],
        "ingredients": ["Retinol / Retinoids", "Peptides", "Hyaluronic Acid", "Vitamin C & E", "Niacinamide", "Ceramides"],
        "avoid": ["Smoking (breaks down collagen rapidly)", "UV exposure without SPF", "Harsh stripping cleansers"],
        "tips": [
            "🏆 Daily SPF is the #1 proven anti-aging tool",
            "🌙 Start retinol slowly (2–3×/week) to avoid irritation",
            "😴 Sleep on your back — pillow friction deepens lines over time",
            "💧 Dehydrated skin shows wrinkles far more visibly — stay hydrated",
        ],
    },
}

def get_severity(conf: float):
    """Return (label, css_class, emoji) based on confidence score."""
    if conf < 0.60:
        return "Mild", "sev-mild", "🟢"
    elif conf < 0.80:
        return "Moderate", "sev-moderate", "🟡"
    else:
        return "Severe", "sev-severe", "🔴"

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "analysis_done": False,
    "condition": None,
    "confidence": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Upload + Result columns ───────────────────────────────────────────────────
upload_col, result_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown('<p class="section-title">📸 Upload Skin Image</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Upload a clear, well-lit photo of your face or skin area",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="upload-hint">📂 Drop a <b>JPG / PNG</b> image here<br>'
            '<span style="font-size:.85rem;opacity:.7">Supports face photos and close-up skin images</span></div>',
            unsafe_allow_html=True,
        )
    else:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Skin", type="primary", use_container_width=True):
            with st.spinner("Analyzing skin conditions…"):
                # Save temporarily so the original function can read a file path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    pil_img.convert("RGB").save(tmp.name)
                    tmp_path = tmp.name

                # ── ORIGINAL PREDICTION FUNCTION ──────────────────────────
                condition, confidence = predict_skin_condition(pil_img)
                # ──────────────────────────────────────────────────────────

                os.unlink(tmp_path)

            st.session_state.condition     = condition
            st.session_state.confidence    = confidence
            st.session_state.analysis_done = True
            st.session_state.chat_history  = []
            st.success("✅ Analysis complete!")

with result_col:
    st.markdown('<p class="section-title">🧬 Diagnosis Results</p>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.markdown(
            '<div class="upload-hint" style="border-color:#a855f7;background:#faf5ff;">'
            '🩺 Your diagnosis will appear here after analysis.</div>',
            unsafe_allow_html=True,
        )
    else:
        cond        = st.session_state.condition
        conf        = st.session_state.confidence
        sev, sev_cls, sev_icon = get_severity(conf)
        info        = SKINCARE_INFO[cond]

        st.markdown(f"""
        <div class="result-card">
            <h2>{info['icon']} {cond}</h2>
            <p><strong>Confidence Score:</strong> {conf*100:.1f}%</p>
            <p><strong>Severity:</strong> <span class="{sev_cls}">{sev_icon} {sev}</span></p>
            <p style="margin-top:.8rem;color:#555;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Confidence: {conf*100:.1f}%**")
        st.progress(float(conf))

        # All class probabilities chart
        with st.expander("📊 Show all class probabilities"):

                img_arr = pil_img.convert("RGB")
                img_arr = img_arr.resize((256, 256))
                img_arr = image.img_to_array(img_arr)
                img_arr = img_arr / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                probs = model.predict(img_arr, verbose=0)[0]

                import pandas as pd

                prob_df = pd.DataFrame({
                    "Condition": classes,
                    "Probability (%)": (probs * 100).round(2)
                })

                st.bar_chart(prob_df.set_index("Condition"))
# ── Recommendations ───────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    st.markdown("---")
    cond = st.session_state.condition
    info = SKINCARE_INFO[cond]

    st.markdown(f'<p class="section-title">💆 Personalized Recommendations — {cond}</p>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3, gap="medium")

    with r1:
        st.markdown("#### 📅 Daily Skincare Routine")
        for step in info["routine"]:
            st.markdown(f'<div class="rec-card">{step}</div>', unsafe_allow_html=True)

    with r2:
        st.markdown("#### ✅ Key Ingredients")
        for ing in info["ingredients"]:
            st.markdown(f'<div class="rec-card-purple">✔ {ing}</div>', unsafe_allow_html=True)

        st.markdown("#### ❌ What to Avoid")
        for av in info["avoid"]:
            st.markdown(f'<div class="rec-card-pink">✘ {av}</div>', unsafe_allow_html=True)

    with r3:
        st.markdown("#### 💡 Pro Tips")
        for tip in info["tips"]:
            st.markdown(f'<div class="rec-card">{tip}</div>', unsafe_allow_html=True)

# ── AI Chatbot ────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    st.markdown("---")
    st.markdown(f'<p class="section-title">🤖 AI Skincare Assistant — Ask about your {st.session_state.condition}</p>', unsafe_allow_html=True)

    # Suggested quick-start questions
    if not st.session_state.chat_history:
        st.markdown("**💬 Quick questions to get started:**")
        suggestions = [
            f"What skincare routine should I follow for {st.session_state.condition}?",
            f"What are the best products for {st.session_state.condition}?",
            "How long until I see improvements?",
            "Are there any natural or home remedies I can try?",
        ]
        btn_cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if btn_cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                with st.spinner("Thinking…"):
                    # ── ORIGINAL CHATBOT FUNCTION ──────────────────────────
                    try:
                        reply = ask_llm(st.session_state.condition, st.session_state.confidence, q)
                    except Exception as e:
                        reply = (
                            f"⚠️ Could not reach Groq: {e}\n\n"
                            
                        )
                    # ──────────────────────────────────────────────────────
                st.session_state.chat_history.append({"role": "user",      "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-label" style="text-align:right;">You</div>'
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-label">🤖 AI Assistant</div>'
                f'<div class="chat-bot">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Free-text input
    user_msg = st.chat_input(f"Ask anything about {st.session_state.condition}…")
    if user_msg:
        with st.spinner("Thinking…"):
            # ── ORIGINAL CHATBOT FUNCTION ──────────────────────────────────
            try:
                reply = ask_llm(st.session_state.condition, st.session_state.confidence, user_msg)
            except Exception as e:
                reply = (
                    f"⚠️ Could not reach Groq: {e}\n\n"
                    "Make sure Groq is running and you have a valid API key."
                )
            # ──────────────────────────────────────────────────────────────
        st.session_state.chat_history.append({"role": "user",      "content": user_msg})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:.85rem;'>"
    "AI Skin Analyzer · Powered by VGG16 + Llama 3 · "
    "For educational purposes only — not a substitute for professional dermatological advice."
    "</p>",
    unsafe_allow_html=True,
)
