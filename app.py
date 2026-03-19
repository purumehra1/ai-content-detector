"""
DeepFake Guardian — Streamlit Web Application
─────────────────────────────────────────────
Hybrid multi-modal deepfake detection with explainable AI reasoning.
Upload a video → get REAL/FAKE verdict with engine-level breakdown.
"""
import streamlit as st
import tempfile
import os
import time
import json

st.set_page_config(
    page_title="DeepFake Guardian",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .verdict-fake {
        background: linear-gradient(135deg, #3d0f0f, #5c1a1a);
        border: 2px solid #ff4444;
        border-radius: 16px; padding: 24px; text-align: center;
        color: #ff8888; font-size: 2rem; font-weight: 800;
    }
    .verdict-real {
        background: linear-gradient(135deg, #0f3d1f, #1a5c2a);
        border: 2px solid #44ff88;
        border-radius: 16px; padding: 24px; text-align: center;
        color: #88ffaa; font-size: 2rem; font-weight: 800;
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, #3d3000, #5c4700);
        border: 2px solid #ffcc00;
        border-radius: 16px; padding: 24px; text-align: center;
        color: #ffdd66; font-size: 2rem; font-weight: 800;
    }
    .engine-card {
        background: #1a1f2e; border-radius: 10px;
        padding: 14px 18px; margin: 6px 0;
        border-left: 4px solid #334;
    }
    .engine-card.bad  { border-left-color: #ff4444; }
    .engine-card.warn { border-left-color: #ffcc00; }
    .engine-card.good { border-left-color: #44ff88; }
    .violation-item {
        background: #1f1520; border-radius: 6px;
        padding: 8px 12px; margin: 4px 0;
        font-size: 0.9rem; color: #ffb3b3;
        border-left: 3px solid #cc3333;
    }
    .info-badge {
        display: inline-block; border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def score_to_class(score: float) -> str:
    if score >= 0.65: return "bad"
    if score >= 0.40: return "warn"
    return "good"


def score_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#ff4444" if score > 0.6 else "#ffcc00" if score > 0.4 else "#44ff88"
    return f"""
    <div style="background:#222;border-radius:6px;height:10px;margin:6px 0;">
        <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;"></div>
    </div>
    <div style="font-size:0.8rem;color:#888;text-align:right;">{pct}% fake probability</div>
    """


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("🛡️ DeepFake Guardian")
    st.caption(
        "Hybrid multi-modal deepfake detector — behavioral analysis instead of just looking for artifacts. "
        "7 engines analyze motion physics, teeth consistency, audio sync, causal rules, and more."
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("v2.0 — Causal AI")

with st.expander("ℹ️ How it works — 7 Detection Engines", expanded=False):
    st.markdown("""
| Engine | What It Detects | Weight |
|---|---|---|
| 🧠 **CNN + GRU** | Visual deepfake artifacts + temporal inconsistency | 40% |
| 🏃 **Biological Motion** | Unnatural velocity, jerk, freeze phases, teleport motion | 20% |
| 🦷 **Teeth Consistency** | GAN flickering, structural changes across frames | 15% |
| 🤚 **Hand / Finger** | Wrong finger count, impossible joint angles | 15% |
| 🔊 **Audio-Visual Sync** | Speech-to-lip correlation, onset delay, dubbing | 5% |
| ⚖️ **Causal Rules** | 8 natural cause-effect laws (speech→jaw, pause→blink…) | 5% |
| 🔬 **Active Stability** | Perturbation sensitivity (real video degrades gracefully) | modifier |

**Core insight**: Instead of asking "does this look fake?", the system asks "does this behave like a real human?"
""")

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a video to analyze",
    type=["mp4", "mov", "avi", "mkv", "webm", "flv"],
    help="Supports MP4, MOV, AVI, MKV, WebM, FLV. Max ~500MB.",
)

if uploaded:
    # Save to temp file
    ext = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(tmp_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_frames = st.slider("Max frames to analyze", 20, 120, 60, 10)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_hands = st.checkbox("Enable Hand Analysis", value=True)
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Video", type="primary", use_container_width=True)

    if analyze_btn:
        progress_area = st.empty()
        status_text = st.empty()

        with st.spinner("Loading detection engines and analyzing..."):
            try:
                from deepfake_detector import DeepFakeDetector

                progress_area.progress(0.05, "Extracting frames and audio...")
                t0 = time.time()

                detector = DeepFakeDetector(max_frames=max_frames, verbose=False)
                result = detector.analyze(tmp_path)

                elapsed = time.time() - t0
                progress_area.progress(1.0, f"Done in {elapsed:.1f}s")
                time.sleep(0.3)
                progress_area.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                os.unlink(tmp_path)
                st.stop()

        st.divider()

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict_class = {
            "FAKE": "verdict-fake",
            "REAL": "verdict-real",
            "SUSPICIOUS": "verdict-suspicious",
        }.get(result.label, "verdict-suspicious")

        icon = {"FAKE": "🤖", "REAL": "✅", "SUSPICIOUS": "⚠️"}.get(result.label, "❓")

        st.markdown(f"""
        <div class="{verdict_class}">
            {icon} {result.label} &nbsp;·&nbsp; {result.confidence_pct:.0f}% confidence
            <div style="font-size:1rem;margin-top:8px;opacity:0.7;">
                Final Score: {result.final_score:.4f} &nbsp;|&nbsp; {result.confidence} confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"> 💬 **{result.summary}**")

        st.divider()

        # ── Engine Breakdown ──────────────────────────────────────────────────
        col_engines, col_violations = st.columns([1, 1])

        with col_engines:
            st.subheader("🔬 Engine Scores")
            for engine, score in result.engine_scores.items():
                if engine == "FINAL":
                    continue
                css_class = score_to_class(float(score))
                st.markdown(f"""
                <div class="engine-card {css_class}">
                    <strong>{engine}</strong>
                    {score_bar(float(score))}
                </div>
                """, unsafe_allow_html=True)

        with col_violations:
            st.subheader(f"⚠️ Violations ({len(result.all_violations)})")
            if result.all_violations:
                for v in result.all_violations:
                    st.markdown(f'<div class="violation-item">⚠ {v}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No behavioral violations detected — video appears natural.")

        st.divider()

        # ── Video Info ────────────────────────────────────────────────────────
        info = result.__dict__.get("video_info", {})
        if info:
            st.subheader("📹 Video Information")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Resolution", f"{info.get('width', '?')}×{info.get('height', '?')}")
            c2.metric("FPS", f"{info.get('fps', '?'):.1f}")
            c3.metric("Duration", f"{info.get('duration', 0):.1f}s")
            c4.metric("Analysis Time", f"{result.__dict__.get('elapsed_seconds', 0):.1f}s")

        # ── JSON Report ───────────────────────────────────────────────────────
        with st.expander("📄 Full JSON Report", expanded=False):
            report = {
                "verdict": result.label,
                "confidence": result.confidence,
                "confidence_pct": result.confidence_pct,
                "final_score": result.final_score,
                "engine_scores": result.engine_scores,
                "violations": result.all_violations,
                "summary": result.summary,
                "video_info": info,
                "analysis_time_s": result.__dict__.get("elapsed_seconds"),
            }
            st.json(report)
            st.download_button(
                "⬇️ Download Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"deepfake_report_{os.path.basename(uploaded.name)}.json",
                mime="application/json",
            )

    os.unlink(tmp_path)

else:
    # Demo info
    st.markdown("""
    ### 🚀 Upload a video to start detection

    **What this system detects:**
    - 🎭 Face-swapped deepfakes (FaceSwap, DeepFaceLab, SimSwap)
    - 🤖 Fully AI-generated videos (Sora, Runway, Stable Video)
    - 🎙️ Voice-dubbed content with face animation (talking head models)
    - 📱 Highly compressed or re-encoded deepfakes (resistant to compression artifacts)

    **What makes this different:**
    > Traditional detectors look for GAN artifacts that disappear in high-quality outputs.
    > This system instead verifies *behavioral physics* — motion, causality, anatomy — that real humans always obey.
    """)

st.divider()
st.caption("DeepFake Guardian · Built by [Puru Mehra](https://github.com/purumehra1) · "
           "Upgraded from ai-content-detector · Nature-Aligned Causal Consistency Architecture")
