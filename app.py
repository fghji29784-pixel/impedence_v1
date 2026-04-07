"""
app.py — DCIM Battery Analyzer — Streamlit entry point.

Run with:
    streamlit run app.py

Structure:
  - Page config + CSS
  - Session state initialisation
  - Sidebar (file upload, settings)
  - Analysis pipeline (load → detect → fit → Nyquist)
  - Tab dispatch → views.py
"""

from __future__ import annotations

import traceback

import streamlit as st

from loader import load_charge_data, load_eis_data
from preprocessor import (
    find_p0_p1_p2, calculate_Rs, prepare_fit_data, detect_I_set,
    prepare_joint_fit_data,
    find_relaxation_start, prepare_relaxation_data,
)
from models import (
    fit_parameters, compute_nyquist,
    voltage_response_2rc, voltage_response_2rc_warburg, voltage_response_1rc,
)
from sidebar import (
    render_cell_selector,
    render_file_upload,
    render_current_unit,
    render_model_selector,
    render_manual_range,
    render_fit_engine,
)
from views import (
    render_tab_raw,
    render_tab_fit,
    render_tab_nyquist,
    render_tab_eis,
    render_tab_diag,
    render_tab_export,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="DCIM Battery Analyzer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
/* ─── 전체 폰트/배경 ─── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
}

/* ─── 사이드바 스타일 ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1C3A5E 100%);
}
[data-testid="stSidebar"] * { color: #E0EAF5 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #64C8E8 !important;
    border-bottom: 1px solid #2A5080;
    padding-bottom: 0.3rem;
    margin-top: 1rem;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #A8C8E0 !important;
    font-size: 0.82rem;
}

/* ─── 스텝 뱃지 ─── */
.step-badge {
    display: inline-block;
    background: #1C7293;
    color: white;
    border-radius: 50%;
    width: 22px; height: 22px;
    text-align: center; line-height: 22px;
    font-size: 0.78rem; font-weight: bold;
    margin-right: 6px;
}

/* ─── 메인 헤더 배너 ─── */
.main-banner {
    background: linear-gradient(90deg, #065A82, #1C7293);
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.2rem;
    color: white;
}
.main-banner h2 { margin: 0; font-size: 1.4rem; }
.main-banner p  { margin: 0.3rem 0 0; opacity: 0.88; font-size: 0.88rem; }

/* ─── 탭 설명 박스 ─── */
.tab-desc {
    background: #EBF5FB;
    border-left: 4px solid #1C7293;
    padding: 0.55rem 0.9rem;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #1A3A50;
    margin-bottom: 0.8rem;
}

/* ─── 진단 결과 카드 ─── */
.diag-card {
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.87rem;
    line-height: 1.5;
}
.diag-ok      { background: #E8F8F0; border-left: 4px solid #27AE60; }
.diag-warn    { background: #FEF9E7; border-left: 4px solid #F39C12; }
.diag-error   { background: #FDEDEC; border-left: 4px solid #E74C3C; }
.diag-info    { background: #EBF5FB; border-left: 4px solid #2980B9; }
.diag-card strong { display: block; margin-bottom: 0.2rem; }

/* ─── 파라미터 의미 테이블 ─── */
.param-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    margin-top: 0.5rem;
}
.param-table th {
    background: #1C7293;
    color: white;
    padding: 6px 10px;
    text-align: left;
}
.param-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #E0EAF5;
}
.param-table tr:nth-child(even) td { background: #F0F7FC; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────

_STATE_KEYS = [
    "df_charge", "df_eis",
    "idx_p0", "idx_p1", "idx_p2",
    "Rs", "I_set",
    "t_fit", "V_fit", "V_pred", "Vp2", "dt",
    "fit_result",
    "re_z", "neg_im_z",
    "cell_key", "nominal_cap_ah",
    "model_choice",
    "eis_fit_results",
    "Rs_dcim_2wire",
    "t_ramp", "I_ramp", "V_ramp",
    "t_relax", "V_relax", "V_relax0", "idx_relax_start",
]
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding: 0.6rem 0 0.4rem;">'
        '<span style="font-size:2rem;">🔋</span>'
        '<div style="font-size:1.05rem; font-weight:700; color:#64C8E8; margin-top:0.2rem;">DCIM Analyzer</div>'
        '<div style="font-size:0.75rem; color:#7AAEC8;">DC Impedance Measurement</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<span class="step-badge">1</span> **셀 타입 선택**', unsafe_allow_html=True)
    cell_key, cell_preset = render_cell_selector()

    st.markdown("---")

    st.markdown('<span class="step-badge">2</span> **데이터 파일 업로드**', unsafe_allow_html=True)
    charge_file, eis_file = render_file_upload()

    st.markdown("---")

    st.markdown('<span class="step-badge">3</span> **분석 설정**', unsafe_allow_html=True)
    st.subheader("⚙️ 기본 설정")
    current_unit = render_current_unit()
    model_choice = render_model_selector()

    st.markdown("---")

    st.markdown('<span class="step-badge">4</span> **고급 옵션** *(선택)*', unsafe_allow_html=True)
    st.subheader("🔧 고급 설정")
    p2_override, window_s, relax_window_s = render_manual_range(default_window=cell_preset["fit_window_s"])
    use_lmfit = render_fit_engine()

    st.markdown("---")
    run_button = st.button("▶  분석 실행", type="primary", width='stretch')

    with st.expander("💡 처음 사용하시나요?"):
        st.markdown("""
**사용 순서:**
1. 셀 타입 선택
2. 충전 데이터 파일 업로드
3. 전류 단위 확인 (보통 mA)
4. **▶ 분석 실행** 클릭
5. 결과 탭에서 확인

**파일 형식:**
- BioLogic `.mpt` (자동 파싱)
- 컬럼: 시간(s), 전압(V), 전류(mA/A)

**문제가 생기면:**
- Rs 이상 → 전류 단위 확인
- R² < 0.95 → 피팅 창 조정
- 오류 발생 → 에러 메시지 확인
        """)


# ──────────────────────────────────────────────
# Main area — 배너 + 탭
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-banner">
  <h2>🔋 DCIM Battery Analyzer</h2>
  <p>
    DC 충전 데이터 → 등가회로 파라미터(Rs, R1, C1, R2, C2) 추출 → 나이퀴스트 플롯 재현 &nbsp;|&nbsp;
    EIS 장비 없이 배터리 임피던스 측정
  </p>
</div>
""", unsafe_allow_html=True)

tab_raw, tab_fit, tab_nyquist, tab_eis, tab_diag, tab_export = st.tabs([
    "📈 Raw Data", "🔧 Fit Result", "🔵 Nyquist Plot", "🔬 EIS 피팅", "🏥 셀 진단", "📥 Export"
])


# ──────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────

if run_button:
    if charge_file is None:
        st.error("❌ 충전 데이터 파일을 먼저 업로드하세요.")
        st.stop()

    for k in _STATE_KEYS:
        st.session_state[k] = None

    st.session_state.cell_key = cell_key

    try:
        with st.spinner("충전 데이터 로딩 중…"):
            charge_file.seek(0)
            df = load_charge_data(charge_file, current_unit=current_unit)
            st.session_state.df_charge = df

        if eis_file is not None:
            with st.spinner("EIS 데이터 로딩 중…"):
                eis_file.seek(0)
                df_eis = load_eis_data(eis_file)
                st.session_state.df_eis = df_eis

        with st.spinner("p0 / p1 / p2 탐지 중…"):
            I_set = detect_I_set(df)
            idx_p0, idx_p1, idx_p2 = find_p0_p1_p2(df, I_set)
            if p2_override is not None:
                idx_p2 = df.index[p2_override]
            st.session_state.I_set  = I_set
            st.session_state.idx_p0 = idx_p0
            st.session_state.idx_p1 = idx_p1
            st.session_state.idx_p2 = idx_p2

        with st.spinner("Rs 계산 중 (ΔV/ΔI)…"):
            Rs = calculate_Rs(df, idx_p0, idx_p1)
            st.session_state.Rs           = Rs
            st.session_state.Rs_dcim_2wire = Rs   # always store 2-wire estimate

        with st.spinner("피팅 데이터 준비 중…"):
            t_fit, V_fit, Vp2, dt = prepare_fit_data(df, idx_p2, window_s=window_s)
            st.session_state.t_fit = t_fit
            st.session_state.V_fit = V_fit
            st.session_state.Vp2   = Vp2
            st.session_state.dt    = dt

            if model_choice == "joint_warburg":
                t_ramp, I_ramp, V_ramp, _, _, V0, _ = prepare_joint_fit_data(
                    df, idx_p0, idx_p2, window_s=window_s,
                )
                st.session_state.t_ramp = t_ramp
                st.session_state.I_ramp = I_ramp
                st.session_state.V_ramp = V_ramp
            else:
                V0 = None

            if model_choice == "relaxation":
                idx_relax = find_relaxation_start(df, I_set, search_after_idx=idx_p2)
                if idx_relax is not None:
                    t_relax, V_relax, V_relax0 = prepare_relaxation_data(
                        df, idx_relax, window_s=relax_window_s,
                    )
                    st.session_state.t_relax       = t_relax
                    st.session_state.V_relax        = V_relax
                    st.session_state.V_relax0       = V_relax0
                    st.session_state.idx_relax_start = idx_relax
                else:
                    st.warning(
                        "⚠️ **이완 구간을 찾을 수 없습니다.** 데이터에 전류 차단 구간이 없습니다.\n"
                        "Relaxation 모델 대신 Extended Randles로 피팅합니다."
                    )
                    model_choice = "extended"

        with st.spinner("등가회로 파라미터 피팅 중…"):
            _t_relax = st.session_state.get("t_relax") if model_choice == "relaxation" else None
            _V_relax = st.session_state.get("V_relax") if model_choice == "relaxation" else None
            _V_relax0 = st.session_state.get("V_relax0") if model_choice == "relaxation" else None
            result = fit_parameters(
                t_fit, V_fit,
                Rs=Rs,
                I=I_set,
                Vp2=Vp2,
                model=model_choice,
                use_lmfit=use_lmfit,
                cell_preset=cell_preset,
                t_ramp=st.session_state.get("t_ramp") if model_choice == "joint_warburg" else None,
                I_ramp=st.session_state.get("I_ramp") if model_choice == "joint_warburg" else None,
                V_ramp=st.session_state.get("V_ramp") if model_choice == "joint_warburg" else None,
                V0=V0,
                t_relax=_t_relax,
                V_relax=_V_relax,
                V_relax0=_V_relax0,
            )
            if model_choice == "joint_warburg":
                st.session_state.Rs = result.Rs
            st.session_state.fit_result = result
            st.session_state.nominal_cap_ah = cell_preset.get("nominal_capacity_ah")
            st.session_state.model_choice = model_choice

        if not result.converged:
            st.warning(
                "⚠️ 피팅이 수렴하지 않았습니다. 초기값(p0)을 결과로 사용합니다.\n\n"
                "**개선 방법:** 셀 타입 확인 → 피팅 창 조정 → lmfit 엔진 전환 → p2 수동 지정"
            )

        with st.spinner("나이퀴스트 곡선 계산 중…"):
            if model_choice == "simple":
                V_pred = voltage_response_1rc(t_fit, result.R1, result.C1, Vp2, I_set)
            elif model_choice in ("warburg", "joint_warburg", "relaxation"):
                V_pred = voltage_response_2rc_warburg(
                    t_fit, result.R1, result.C1, result.R2, result.C2,
                    result.sigma_W, Vp2, I_set,
                )
            else:
                V_pred = voltage_response_2rc(
                    t_fit, result.R1, result.C1, result.R2, result.C2, Vp2, I_set
                )
            st.session_state.V_pred = V_pred

            re_z, neg_im_z = compute_nyquist(
                result.Rs, result.R1, result.C1, result.R2, result.C2,
                sigma_W=result.sigma_W,
            )
            st.session_state.re_z     = re_z
            st.session_state.neg_im_z = neg_im_z

        st.success("✅ 분석 완료!")

    except Exception as exc:
        err_str = str(exc).lower()
        if "cannot identify time" in err_str or "cannot identify" in err_str:
            st.error(
                f"❌ 컬럼 인식 실패: {exc}\n\n"
                "**해결책:** 파일에 시간(time/s), 전압(Ewe/V 또는 Voltage/V), "
                "전류(I/mA 또는 I/A) 컬럼이 있는지 확인하세요."
            )
        elif "cannot decode" in err_str:
            st.error(
                f"❌ 파일 인코딩 오류: {exc}\n\n"
                "**해결책:** UTF-8, CP949, Latin-1 이외의 인코딩으로 저장된 파일입니다. "
                "파일을 UTF-8로 다시 저장하거나 BioLogic 원본 .mpt 파일을 사용하세요."
            )
        elif "current never settles" in err_str or "cannot find p2" in err_str:
            st.error(
                f"❌ p2 탐지 실패: {exc}\n\n"
                "**해결책:**\n"
                "① 사이드바 '고급 설정'에서 **p2 수동 지정**을 활성화하세요.\n"
                "② 전류 단위(mA/A)가 올바른지 확인하세요.\n"
                "③ 데이터에 충분한 정상 전류 구간이 있는지 확인하세요."
            )
        elif "di ≈ 0" in err_str or "cannot compute rs" in err_str:
            st.error(
                f"❌ Rs 계산 실패 (ΔI ≈ 0): {exc}\n\n"
                "**해결책:** 전류 단위 설정을 확인하세요. "
                "파일의 전류가 mA 단위인데 A로 설정했거나 그 반대일 수 있습니다."
            )
        else:
            st.error(f"❌ 분석 중 오류 발생: {exc}")

        with st.expander("🔍 상세 오류 내용 (개발자용)"):
            st.code(traceback.format_exc())


# ──────────────────────────────────────────────
# Tab rendering — delegated to views.py
# ──────────────────────────────────────────────

with tab_raw:
    render_tab_raw()

with tab_fit:
    render_tab_fit()

with tab_nyquist:
    render_tab_nyquist()

with tab_eis:
    render_tab_eis()

with tab_diag:
    render_tab_diag()

with tab_export:
    render_tab_export()
