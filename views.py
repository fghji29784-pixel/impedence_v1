"""
views.py — Streamlit 탭 렌더링 함수 모음.

app.py에서 6개 탭의 렌더링 로직을 분리한 모듈.
각 함수는 st.session_state를 읽어 탭 UI를 그린다.

Changes from DCIM_claude:
  - app.py에서 분리 (app.py 1318줄 → ~250줄 + views.py ~750줄)
  - 진단 탭 자동 가져오기 버그 수정:
      "result" 키 (존재하지 않음) → "fit_result" 키 사용
      result.get("Rs") (dict API) → fit_res.Rs (FitResult 속성)
      Ω → mΩ 변환 명시적 추가
  - DCIMParams 생성 시 Ω 단위로 전달 (UI 입력값 mΩ → /1000)
  - nyquist_from_params 반환값이 Ω이므로 mΩ 표시 시 *1000
  - diagnostics R_total 표시: Ω → mΩ 변환
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from models import CELL_EXPECTED_RANGES
from plotter import plot_raw_data, plot_fit_result, plot_nyquist, plot_eis_fit
from exporter import export_results_excel, export_report_text
from eis_fitter import compare_dcim_eis, MODELS as EIS_MODELS
from diagnostics import CellDiagnostics, DCIMParams, nyquist_from_params


# ──────────────────────────────────────────────
# Formation defect analysis
# ──────────────────────────────────────────────

def analyze_formation_defect(result, cell_key: str, I_set: float, nominal_cap_ah) -> list[tuple]:
    """
    DCIM 파라미터를 활성화 공정 불량 관점에서 해석합니다.

    Returns list of (level, title, body)
      level: "ok" | "warn" | "error" | "info"
    """
    findings = []
    ranges = CELL_EXPECTED_RANGES.get(cell_key, CELL_EXPECTED_RANGES["custom"])
    rs_mohm = result.Rs * 1000
    r1_mohm = result.R1 * 1000
    r2_mohm = result.R2 * 1000
    rs_min, rs_max = ranges["Rs_mohm"]
    r1_min, r1_max = ranges["R1_mohm"]
    r2_min, r2_max = ranges["R2_mohm"]

    # ── Rs 평가 ──────────────────────────────────────────────────────────
    if cell_key == "custom":
        findings.append(("info", "ℹ️ Custom 셀 — 정상 범위 미적용",
            "셀 타입을 선택하면 Rs/R1/R2 정상 범위 비교가 자동으로 활성화됩니다."))
    elif rs_mohm < rs_min * 0.60:
        findings.append(("error", "⚡ 저전압 불량 가능성 (Rs 비정상 저하)",
            f"Rs = {rs_mohm:.2f} mΩ (정상 하한 {rs_min:.0f} mΩ).\n"
            "내부 미세단락(micro-short)이 형성되면 병렬 전류 경로가 생겨\n"
            "순간 전압 강하(ΔV)가 작아집니다 → Rs가 낮게 계산됩니다.\n"
            "📋 권장 조치: OCV 모니터링 및 자가방전 시험 실시."))
    elif rs_mohm > rs_max * 1.50:
        findings.append(("error", "💧 함침 불량 의심 (Rs 과대)",
            f"Rs = {rs_mohm:.2f} mΩ (정상 상한 {rs_max:.0f} mΩ).\n"
            "전해액이 전극 기공을 충분히 채우지 못하면 이온 전도 경로가 막혀\n"
            "전해질 저항(Rs)이 높게 측정됩니다.\n"
            "📋 권장 조치: 함침 시간 연장 또는 추가 진공 함침 검토."))
    elif rs_mohm > rs_max * 1.15:
        findings.append(("warn", "⚠️ Rs 경미한 상승",
            f"Rs = {rs_mohm:.2f} mΩ (정상 상한 {rs_max:.0f} mΩ 소폭 초과).\n"
            "함침 진행 중이거나 초기 사이클에서 SEI가 아직 안정화되지 않은 상태일 수 있습니다.\n"
            "추가 사이클 후 재측정을 권장합니다."))
    else:
        findings.append(("ok", "✅ Rs 정상 범위",
            f"Rs = {rs_mohm:.2f} mΩ — {ranges['Rs_mohm']} mΩ 정상 범위 내."))

    # ── R2 평가 (함침 불량 2차 지표) ─────────────────────────────────────
    if cell_key != "custom" and result.R2 > 0:
        if r2_mohm > r2_max * 2.0:
            findings.append(("warn", "💧 함침 불량 보조 지표 (R₂ 과대)",
                f"R₂ = {r2_mohm:.2f} mΩ (정상 상한 {r2_max:.0f} mΩ의 2배 초과).\n"
                "고체상 확산 저항이 높습니다. 전해액이 전극 입자 내부 기공까지\n"
                "충분히 침투하지 못했을 가능성이 있습니다."))
        elif r2_mohm < r2_min * 0.3:
            findings.append(("warn", "⚠️ R₂ 비정상 저하",
                f"R₂ = {r2_mohm:.2f} mΩ — 예상보다 매우 낮습니다.\n"
                "피팅 수렴이 비정상적일 수 있습니다. 피팅 창 또는 초기값을 확인하세요."))

    # ── 피팅 품질 평가 ────────────────────────────────────────────────────
    if result.r2 < 0.95:
        findings.append(("error", "❌ 피팅 품질 불량 (R² < 0.95)",
            f"R² = {result.r2:.4f}, RMSE = {result.rmse_mv:.2f} mV.\n"
            "파라미터 신뢰도가 낮습니다. 가능한 원인:\n"
            "① p2 자동 탐지 오류 → 수동 지정 시도\n"
            "② 피팅 창이 너무 짧음 → window 늘리기\n"
            "③ 4680/4695: 초기값이 잘못됨 → 셀 타입 확인\n"
            "④ 데이터에 노이즈/이상점 존재"))
    elif result.r2 < 0.98:
        findings.append(("warn", "⚠️ 피팅 품질 주의 (R² 0.95–0.98)",
            f"R² = {result.r2:.4f}. 결과 해석 시 주의가 필요합니다.\n"
            "lmfit 엔진 또는 피팅 창 조정으로 개선을 시도해 보세요."))
    else:
        findings.append(("ok", "✅ 피팅 품질 양호",
            f"R² = {result.r2:.4f}, RMSE = {result.rmse_mv:.2f} mV — 신뢰할 수 있는 결과입니다."))

    # ── C-rate 정보 ──────────────────────────────────────────────────────
    if nominal_cap_ah and nominal_cap_ah > 0 and I_set and I_set > 0:
        c_rate = I_set / nominal_cap_ah
        if c_rate <= 0.1:
            findings.append(("info",
                f"📊 측정 C-rate: {c_rate:.4f}C  ({I_set * 1000:.1f} mA)",
                "DCIM 권장 범위(0.02C–0.1C) 내에 있습니다."))
        else:
            findings.append(("warn",
                f"⚠️ 높은 C-rate: {c_rate:.4f}C  ({I_set * 1000:.1f} mA)",
                "0.1C 초과 전류에서는 분극이 커져 R₂ 추정 오차가 증가할 수 있습니다.\n"
                "가능하면 0.02C–0.05C 범위에서 측정을 권장합니다."))

    return findings


def _diag_html(level: str, title: str, body: str) -> str:
    cls = {"ok": "diag-ok", "warn": "diag-warn", "error": "diag-error", "info": "diag-info"}
    body_html = body.replace("\n", "<br>")
    return (
        f'<div class="diag-card {cls.get(level, "diag-info")}">'
        f'<strong>{title}</strong>{body_html}</div>'
    )


# ──────────────────────────────────────────────
# Tab 1: Raw Data
# ──────────────────────────────────────────────

def render_tab_raw() -> None:
    st.markdown(
        '<div class="tab-desc">'
        '업로드된 충전 데이터와 자동으로 탐지된 임계점(p0·p1·p2)을 확인합니다. '
        'p0/p1/p2가 올바르게 탐지되었는지 반드시 확인한 후 다음 탭으로 넘어가세요.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.df_charge is not None and st.session_state.idx_p2 is not None:
        df    = st.session_state.df_charge
        I_set = st.session_state.I_set
        nom   = st.session_state.nominal_cap_ah

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("p0 인덱스", st.session_state.idx_p0,
                    help="전류 인가 직전 안정 상태 마지막 포인트")
        col2.metric("p1 인덱스", st.session_state.idx_p1,
                    help="I_set의 50% 이상 도달 첫 번째 포인트 → Rs 계산에 사용")
        col3.metric("p2 인덱스", st.session_state.idx_p2,
                    help="I_set의 99% 이내로 안정된 첫 번째 포인트 → 피팅 시작점")
        _rs_2wire = st.session_state.get("Rs_dcim_2wire")
        _model = st.session_state.get("model_choice", "extended")
        if _model == "joint_warburg" and _rs_2wire is not None:
            col4.metric("Rs (2-wire ΔV/ΔI)", f"{_rs_2wire * 1000:.3f} mΩ",
                        help="ΔV/ΔI(p0→p1) 추정값. Joint 모델에서는 피팅된 Rs가 Fit Result 탭에 표시됩니다.")
        else:
            col4.metric("Rs", f"{st.session_state.Rs * 1000:.3f} mΩ",
                        help="옴 저항 = ΔV/ΔI (p0→p1)")
        col5.metric("I_set", f"{I_set * 1000:.1f} mA",
                    help="측정 전류 (95th percentile 추정)")

        if nom and nom > 0:
            c_rate = I_set / nom
            col6.metric("C-rate", f"{c_rate:.4f} C",
                        help=f"I_set({I_set*1000:.1f} mA) / 공칭용량({nom:.1f} Ah)")
        else:
            col6.metric("dt", f"{st.session_state.dt * 1000:.3f} ms",
                        help="p2 근처 로컬 샘플링 간격")

        fig = plot_raw_data(
            df,
            st.session_state.idx_p0,
            st.session_state.idx_p1,
            st.session_state.idx_p2,
        )
        st.pyplot(fig)

        with st.expander("📖 p0 / p1 / p2 란 무엇인가?"):
            st.markdown("""
| 포인트 | 정의 | 역할 |
|--------|------|------|
| **p0** | 전류 인가 직전 마지막 안정 상태 | 기준 OCV, 기준 전류 |
| **p1** | \|I_set − I\| / I_set < 50% 첫 번째 점 | Rs 계산용 (`Rs = ΔV/ΔI`) |
| **p2** | \|I_set − I\| / I_set < 1% 첫 번째 점 | 피팅 시작점 (`t=0`, `V=Vp2`) |

**p2가 잘못 탐지된 경우** → 사이드바 '고급 설정'에서 수동 지정하세요.
            """)

        with st.expander("📋 원시 데이터 테이블 (처음 200행)"):
            st.dataframe(df.head(200))
    else:
        st.info("👆 사이드바에서 파일을 업로드하고 **▶ 분석 실행**을 클릭하세요.")
        with st.expander("📖 DCIM 이란?"):
            st.markdown("""
**DCIM (DC Impedance Measurement)** 은 배터리에 일정 전류(DC)를 인가했을 때
나타나는 전압 과도 응답(voltage transient)을 분석하여 배터리 내부 임피던스를
추출하는 방법입니다.

**측정 원리:**
```
전류 스텝 인가
→ 전압 과도 응답 기록 (서브 밀리초 간격)
→ Extended Randles 모델에 커브 피팅
→ Rs, R₁, C₁, R₂, C₂ 추출
→ Z(jω) = Rs + R₁/(1+jωR₁C₁) + R₂/(1+jωR₂C₂)
→ 나이퀴스트 플롯 재현
```

**EIS와의 차이:**
- EIS: AC 신호 스캔, 전용 장비 필요, 수십 분 소요
- DCIM: DC 펄스, 충전기 내장, 수 초 완료
            """)


# ──────────────────────────────────────────────
# Tab 2: Fit Result
# ──────────────────────────────────────────────

def render_tab_fit() -> None:
    st.markdown(
        '<div class="tab-desc">'
        '등가회로 파라미터 피팅 결과와 활성화 공정 불량 진단을 확인합니다. '
        '<b>R² > 0.99</b> 이상이면 신뢰할 수 있는 결과입니다.'
        '</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.fit_result

    if result is not None:
        # 수렴 실패 경고
        if not result.converged:
            st.warning(
                "⚠️ **피팅 수렴 실패** — 표시된 파라미터는 초기 추정값입니다. "
                "물리적으로 해석하지 마세요.\n\n"
                "개선 방법: 셀 타입 확인 / 피팅 창 조정 / lmfit 엔진 전환 / p2 수동 지정"
            )

        _model_c = st.session_state.get("model_choice", "extended")
        _rs_2w   = st.session_state.get("Rs_dcim_2wire")
        if _model_c == "joint_warburg" and _rs_2w is not None:
            st.info(
                f"**Joint Warburg 모델**: Rs는 ramp+CC 전 구간 동시 피팅으로 추출된 값입니다. "
                f"(ΔV/ΔI 2-wire 추정치: {_rs_2w*1000:.3f} mΩ → 피팅된 Rs: {result.Rs*1000:.3f} mΩ)"
            )
        col1, col2, col3, col4, col5 = st.columns(5)
        _rs_help = ("ramp+CC 전 구간 동시 피팅으로 추출된 Rs. τ₁이 짧으면 ΔV/ΔI보다 낮게 나옴."
                    if _model_c == "joint_warburg" else
                    "전해질 저항 + 접촉 저항. 순간 전압 강하(ΔV/ΔI)에서 계산")
        col1.metric("Rs",   f"{result.Rs * 1000:.3f} mΩ", help=_rs_help)
        col2.metric("R₁",   f"{result.R1 * 1000:.3f} mΩ",
                    help="SEI막/전하이동 저항 (빠른 RC)")
        col3.metric("R₂",   f"{result.R2 * 1000:.3f} mΩ",
                    help="고체상 확산 저항 (느린 RC). 함침 불량 시 상승")
        col4.metric("R²",   f"{result.r2:.5f}",
                    delta="양호" if result.r2 >= 0.99 else ("주의" if result.r2 >= 0.95 else "불량"),
                    delta_color="normal" if result.r2 >= 0.99 else "inverse")
        col5.metric("RMSE", f"{result.rmse_mv:.3f} mV",
                    help="피팅 잔차 제곱평균제곱근. 낮을수록 피팅 정확도 높음")

        col_a, col_b = st.columns([1.3, 1])

        with col_a:
            fig = plot_fit_result(
                st.session_state.t_fit,
                st.session_state.V_fit,
                st.session_state.V_pred,
                result,
                Vp2=st.session_state.Vp2,
                I=st.session_state.I_set,
                model=st.session_state.get("model_choice", "extended"),
            )
            st.pyplot(fig)

        with col_b:
            st.markdown("#### 🏭 활성화 공정 불량 진단")
            st.caption(
                "⚠️ 이 진단은 **참고용 가이드라인**입니다. "
                "불량 판정 임계값은 실제 데이터로 반드시 보정하세요."
            )

            cell_k = st.session_state.cell_key or "custom"
            findings = analyze_formation_defect(
                result,
                cell_k,
                st.session_state.I_set,
                st.session_state.nominal_cap_ah,
            )
            for level, title, body in findings:
                st.markdown(_diag_html(level, title, body), unsafe_allow_html=True)

        st.markdown("#### 📊 전체 파라미터")
        with st.expander("파라미터 상세 테이블 열기", expanded=True):
            _params = ["Rs", "R₁", "C₁", "R₂", "C₂", "τ₁", "τ₂", "f₁", "f₂", "R²", "RMSE"]
            _vals = [
                f"{result.Rs * 1000:.4f} mΩ",
                f"{result.R1 * 1000:.4f} mΩ",
                f"{result.C1:.4f} F",
                f"{result.R2 * 1000:.4f} mΩ",
                f"{result.C2:.4f} F",
                f"{result.tau1 * 1000:.3f} ms",
                f"{result.tau2:.4f} s",
                f"{result.f1:.3f} Hz",
                f"{result.f2:.5f} Hz",
                f"{result.r2:.6f}",
                f"{result.rmse_mv:.4f} mV",
            ]
            _errs = [
                "—",
                f"{result.sigma_R1 * 1000:.4f} mΩ",
                f"{result.sigma_C1:.4f} F",
                f"{result.sigma_R2 * 1000:.4f} mΩ",
                f"{result.sigma_C2:.4f} F",
                "—", "—", "—", "—", "—", "—",
            ]
            # Warburg 모델인 경우 σ_W 행 추가
            if getattr(result, "sigma_W", 0.0) > 0.0:
                _params.insert(-2, "σ_W (Warburg)")
                _vals.insert(-2, f"{result.sigma_W * 1000:.4f} mΩ·s⁻¹/²")
                _errs.insert(-2, "—")
                st.info(
                    f"**Warburg 모델**: σ_W = {result.sigma_W*1000:.4f} mΩ·s⁻¹/²\n\n"
                    "√t 항이 고체상 확산 전압을 흡수했으므로 **R₁+R₂ ≈ EIS Rct**에 "
                    "해당합니다. Rs 차이는 측정 방식(DCIM 2-wire vs EIS 4-wire)으로 "
                    "인한 케이블/접촉 저항 포함 여부로 발생합니다."
                )
            param_df = pd.DataFrame({"파라미터": _params, "값": _vals, "±1σ": _errs})
            st.dataframe(param_df, width='stretch', hide_index=True)

        with st.expander("📖 각 파라미터의 물리적 의미"):
            st.markdown("""
<table class="param-table">
<tr><th>파라미터</th><th>물리적 의미</th><th>활성화 공정 관련성</th></tr>
<tr><td><b>Rs</b></td>
    <td>전해질 이온 저항 + 집전체/접촉 저항 + SEI 벌크 저항.<br>
        전류 인가 즉시 나타나는 순간 전압 강하(ΔV/ΔI).</td>
    <td>높으면 → 함침 불량 (전해액 부족)<br>낮으면 → 미세단락 가능성</td></tr>
<tr><td><b>R₁, C₁</b></td>
    <td>SEI막 이온 통과 저항 + 전기이중층 용량.<br>
        빠른 시정수 τ₁ = R₁·C₁ (수 ms ~ 수십 ms).</td>
    <td>Formation 사이클 진행 시 SEI 성장으로 점진적 증가 후 안정화</td></tr>
<tr><td><b>R₂, C₂</b></td>
    <td>고체 전극 내 Li⁺ 확산 저항.<br>
        느린 시정수 τ₂ = R₂·C₂ (수 초 ~ 수백 초).</td>
    <td>높으면 → 함침 불완전 (기공 내 전해액 미침투)</td></tr>
<tr><td><b>τ₁, τ₂</b></td>
    <td>각 RC 회로의 이완 시간상수.<br>
        나이퀴스트 반원 피크 주파수 f = 1/(2π·τ).</td>
    <td>셀 크기·온도에 따라 달라짐</td></tr>
<tr><td><b>R²</b></td>
    <td>피팅 결정계수. 1.0에 가까울수록 모델이 데이터를 잘 설명.</td>
    <td>&gt; 0.99 : 신뢰 가능 / &lt; 0.95 : 파라미터 재확인 필요</td></tr>
</table>
""", unsafe_allow_html=True)

    else:
        st.info("▶ 분석 실행 후 결과가 여기에 표시됩니다.")
        with st.expander("📖 Extended Randles 모델이란?"):
            st.markdown("""
**Extended Randles 회로**: `Rs + R₁||C₁ + R₂||C₂`

**시간영역 응답 (DCIM이 피팅하는 방정식):**
```
V(t) = Vp2 + R₁·I·(1 − e^{−t/τ₁}) + R₂·I·(1 − e^{−t/τ₂})
```

**주파수영역 임피던스 (나이퀴스트 플롯 생성):**
```
Z(jω) = Rs + R₁/(1 + jωR₁C₁) + R₂/(1 + jωR₂C₂)
```
""")


# ──────────────────────────────────────────────
# Tab 3: Nyquist Plot
# ──────────────────────────────────────────────

def render_tab_nyquist() -> None:
    st.markdown(
        '<div class="tab-desc">'
        'DCIM이 재현한 나이퀴스트 플롯입니다. '
        'EIS 파일도 업로드하면 실측값과 직접 비교할 수 있습니다. '
        '<b>반원이 원형</b>으로 보여야 정상입니다 (x축·y축 스케일 동일).'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.re_z is not None:
        # ── EIS Rs: prefer fitted value from best EIS model ───────────────
        eis_rs_fit = None
        eis_fit_res = st.session_state.get("eis_fit_results")
        if eis_fit_res:
            best_conv = next((r for r in eis_fit_res if r.converged), None)
            if best_conv is not None:
                eis_rs_fit = best_conv.params_dict.get("Rs") or best_conv.Rs

        col_plot, col_info = st.columns([1.5, 1])

        with col_plot:
            fig = plot_nyquist(
                st.session_state.re_z,
                st.session_state.neg_im_z,
                eis_df=st.session_state.df_eis,
                result=st.session_state.fit_result,
                eis_rs_fit=eis_rs_fit,
            )
            st.pyplot(fig)

        with col_info:
            st.markdown("#### 📖 나이퀴스트 플롯 읽는 법")
            st.markdown("""
**축 의미:**
- **X축 Re(Z)**: 실수부 (전기저항 성분)
- **Y축 -Im(Z)**: 허수부 음수 (전기용량 성분)

**그래프 특징:**
| 구간 | 의미 |
|------|------|
| 왼쪽 실수축 절편 | **Rs** (전해질/접촉 저항) |
| 첫 번째 반원 지름 | **R₁** (SEI·계면) |
| 두 번째 반원 지름 | **R₂** (확산 저항) |
| 오른쪽 끝 | Rs + R₁ + R₂ (DC 총 저항) |

**EIS-보정 DCIM 곡선 (파란 실선):**
DCIM은 2-wire 측정이라 Rs에 케이블/접촉 저항이 포함됩니다.
EIS Rs(4-wire)와 맞춰 x축 오프셋 보정한 곡선입니다.
아크 **모양(R₁, R₂)** 비교에 사용하세요.
""")

        if st.session_state.df_eis is not None:
            df_eis  = st.session_state.df_eis
            result  = st.session_state.fit_result

            # EIS Rs: fitted > capacitive arc min > full min (우선순위)
            df_eis_cap = df_eis[df_eis["neg_im_z"] >= 0]
            if eis_rs_fit is not None:
                rs_eis     = eis_rs_fit * 1000
                rs_eis_src = "EIS 피팅"
            elif len(df_eis_cap) > 0:
                rs_eis     = float(df_eis_cap["re_z"].min()) * 1000
                rs_eis_src = "capacitive arc 최솟값"
            else:
                rs_eis     = float(df_eis["re_z"].min()) * 1000
                rs_eis_src = "전체 데이터 최솟값"
            rs_dcim = result.Rs * 1000

            # R1+R2: EIS arc width = Re(Z) at peak - Rs
            if len(df_eis_cap) > 0:
                idx_peak   = df_eis_cap["neg_im_z"].idxmax()
                re_at_peak = float(df_eis_cap.loc[idx_peak, "re_z"]) * 1000
                r_arc_eis  = re_at_peak - rs_eis
            else:
                r_arc_eis  = 0.0
            r_arc_dcim = (result.R1 + result.R2) * 1000

            st.subheader("📊 EIS vs DCIM 수치 비교")
            st.caption(f"EIS Rs 출처: {rs_eis_src}")
            cmp_df = pd.DataFrame({
                "파라미터": ["Rs (Ω 저항)", "R₁+R₂ (아크 폭, 근사)"],
                "EIS [mΩ]":  [f"{rs_eis:.3f}",    f"{r_arc_eis:.3f}"],
                "DCIM [mΩ]": [f"{rs_dcim:.3f}",   f"{r_arc_dcim:.3f}"],
                "차이 [mΩ]": [
                    f"{rs_dcim - rs_eis:+.3f}",
                    f"{r_arc_dcim - r_arc_eis:+.3f}",
                ],
                "오차율": [
                    f"{abs(rs_dcim - rs_eis) / rs_eis * 100:.1f}%" if rs_eis else "—",
                    f"{abs(r_arc_dcim - r_arc_eis) / r_arc_eis * 100:.1f}%" if r_arc_eis else "—",
                ],
            })
            st.dataframe(cmp_df, width='stretch', hide_index=True)

            if rs_eis > 0:
                rs_ratio = rs_dcim / rs_eis
                if rs_ratio < 0.5 or rs_ratio > 2.0:
                    st.warning(
                        f"⚠️ **Rs 불일치 경고**: DCIM({rs_dcim:.2f} mΩ) / EIS({rs_eis:.2f} mΩ) "
                        f"= {rs_ratio:.2f}×  (허용 범위: 0.5×~2.0×)\n\n"
                        "주요 원인:\n"
                        "- **2-wire DCIM** vs **4-wire EIS**: 케이블/접촉 저항이 DCIM Rs에 포함\n"
                        "- 측정 온도·SOC 차이\n\n"
                        "Nyquist 플롯의 **EIS-보정 곡선(파란 실선)**은 이 차이를 보정하여\n"
                        "아크 모양 비교에 사용할 수 있습니다."
                    )

        with st.expander("📋 나이퀴스트 데이터 테이블"):
            nyq_df = pd.DataFrame({
                "Re(Z) [mΩ]":  st.session_state.re_z * 1000,
                "-Im(Z) [mΩ]": st.session_state.neg_im_z * 1000,
            })
            st.dataframe(nyq_df, width='stretch')
    else:
        st.info("▶ 분석 실행 후 나이퀴스트 플롯이 여기에 표시됩니다.")


# ──────────────────────────────────────────────
# Tab 4: EIS Fitting
# ──────────────────────────────────────────────

def render_tab_eis() -> None:
    from eis_fitter import fit_eis_all_models

    st.markdown(
        '<div class="tab-desc">'
        'EIS 실측 데이터를 등가회로 모델에 직접 피팅합니다. '
        '4가지 모델(2-RC, 2-RC+CPE, 3-RC, Randles+W)을 비교하여 <b>AIC 기준 최적 회로를 자동 선택</b>합니다.'
        '</div>',
        unsafe_allow_html=True,
    )

    df_eis_tab = st.session_state.df_eis

    if df_eis_tab is None:
        st.info("👆 사이드바에서 **EIS 데이터 파일**을 업로드한 후 **▶ 분석 실행**을 클릭하세요.")
        with st.expander("📖 EIS 피팅이란? (CNLS)"):
            st.markdown("""
**CNLS (Complex Nonlinear Least Squares)** 란 EIS의 복소 임피던스 Z(ω)를
등가회로 모델 함수에 최소제곱법으로 피팅하는 방법입니다.

| 모델 | 파라미터 | 특징 |
|------|---------|------|
| 2-RC | Rs, R₁, C₁, R₂, C₂ (5개) | DCIM 기본 모델, 이상적 반원 |
| 2-RC+CPE | + α₁, α₂ (7개) | 눌린 반원, 비이상적 표면 |
| 3-RC | Rs, R₁-₃, C₁-₃ (7개) | 3개 반원, 대형 셀 |
| Randles+W | Rs, Rct, Cdl, σ (4개) | 저주파 Warburg 확산 |
""")
    else:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_eis_fit = st.button(
                "🔬 EIS 피팅 실행",
                type="primary",
                width='stretch',
            )
        with col_info:
            st.caption(
                f"EIS 데이터 로드 완료: **{len(df_eis_tab)}** 주파수 포인트 | "
                f"주파수 범위: {df_eis_tab['freq'].min():.3g} ~ {df_eis_tab['freq'].max():.3g} Hz"
            )

        if run_eis_fit:
            with st.spinner("4개 등가회로 모델 CNLS 피팅 중…"):
                try:
                    eis_results = fit_eis_all_models(
                        df_eis_tab["freq"].values,
                        df_eis_tab["re_z"].values,
                        df_eis_tab["neg_im_z"].values,
                    )
                    st.session_state.eis_fit_results = eis_results
                    st.success(f"✅ 피팅 완료! 최적 모델: **{eis_results[0].model_label}**  (AIC = {eis_results[0].aic:.2f})")
                except Exception as exc:
                    st.error(f"❌ EIS 피팅 오류: {exc}")
                    with st.expander("상세 오류"):
                        import traceback as _tb
                        st.code(_tb.format_exc())

        eis_results = st.session_state.eis_fit_results

        if eis_results:
            best = eis_results[0]

            st.markdown("#### 📊 모델 비교 (AIC 기준 랭킹)")
            cmp_rows = []
            for rank, r in enumerate(eis_results):
                delta_aic = r.aic - best.aic
                badge = "🥇 최적" if rank == 0 else (f"Δ+{delta_aic:.1f}" if delta_aic < 100 else "❌ 부적합")
                cmp_rows.append({
                    "순위": f"{rank+1}위 {badge}",
                    "모델": r.model_label,
                    "파라미터 수": EIS_MODELS[r.model_name]["n_params"],
                    "AIC": f"{r.aic:.2f}",
                    "R²(합산)": f"{r.r2_total:.5f}" if r.converged else "수렴 실패",
                    "RMSE [mΩ]": f"{r.rmse_mohm:.3f}" if r.converged else "—",
                    "수렴": "✅" if r.converged else "❌",
                })
            st.dataframe(pd.DataFrame(cmp_rows), width='stretch', hide_index=True)
            st.caption("AIC(아카이케 정보기준): 낮을수록 좋음. ΔAIC < 2이면 통계적으로 동등 → 더 단순한 모델 선택 권장.")

            st.markdown("#### 🔍 모델별 상세 결과")
            model_tabs = st.tabs([f"{'🥇 ' if i==0 else ''}{r.model_name}" for i, r in enumerate(eis_results)])

            for tab, r in zip(model_tabs, eis_results):
                with tab:
                    if not r.converged:
                        st.error(f"❌ 수렴 실패: {r.message}")
                        continue

                    col_l, col_r = st.columns([1.3, 1])

                    with col_l:
                        try:
                            fig_eis = plot_eis_fit(r)
                            st.pyplot(fig_eis)
                        except Exception as e:
                            st.warning(f"플롯 오류: {e}")

                    with col_r:
                        st.markdown(f"**{r.model_label}**")
                        st.markdown(EIS_MODELS[r.model_name]["description"])
                        st.markdown("")

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("AIC",  f"{r.aic:.2f}")
                        mc2.metric("R²",   f"{r.r2_total:.5f}")
                        mc3.metric("RMSE", f"{r.rmse_mohm:.3f} mΩ")

                        st.markdown("**파라미터 피팅 결과:**")
                        param_rows = []
                        pd_vals = r.params_dict
                        pd_errs = r.errors_dict
                        for pname in r.param_names:
                            v = pd_vals[pname]
                            e = pd_errs[pname]
                            if pname.startswith("R"):
                                disp_v = f"{v * 1000:.4f} mΩ"
                                disp_e = f"± {e * 1000:.4f} mΩ" if not np.isnan(e) else "—"
                            elif pname.startswith("C") or pname.startswith("Cdl"):
                                disp_v = f"{v:.4f} F"
                                disp_e = f"± {e:.4f} F" if not np.isnan(e) else "—"
                            elif pname.startswith("α"):
                                disp_v = f"{v:.4f}"
                                disp_e = f"± {e:.4f}" if not np.isnan(e) else "—"
                            elif pname.startswith("Q"):
                                disp_v = f"{v:.4e} S·sα"
                                disp_e = f"± {e:.4e}" if not np.isnan(e) else "—"
                            elif pname == "σ":
                                disp_v = f"{v:.4f} Ω·s⁻¹/²"
                                disp_e = f"± {e:.4f}" if not np.isnan(e) else "—"
                            else:
                                disp_v = f"{v:.5g}"
                                disp_e = f"± {e:.3g}" if not np.isnan(e) else "—"
                            param_rows.append({"파라미터": pname, "값": disp_v, "±1σ": disp_e})
                        st.dataframe(pd.DataFrame(param_rows), width='stretch', hide_index=True)

            # DCIM vs EIS 비교
            dcim_res = st.session_state.fit_result
            eis_2rc  = next((r for r in eis_results if r.model_name == "2RC"), None)

            if dcim_res is not None and eis_2rc is not None and eis_2rc.converged:
                st.markdown("#### ⚖️ DCIM vs EIS 피팅 파라미터 비교 (2-RC 모델)")
                comparison = compare_dcim_eis(dcim_res, eis_2rc)
                if comparison:
                    cmp_display = []
                    for row in comparison:
                        pct = row["오차율"]
                        flag = ("✅" if not np.isnan(pct) and pct < 5
                                else ("⚠️" if not np.isnan(pct) and pct < 15
                                      else ("❌" if not np.isnan(pct) else "—")))
                        cmp_display.append({
                            "파라미터": row["파라미터"],
                            "DCIM":    f"{row['DCIM']:.4f} {row['단위']}",
                            "EIS 피팅": f"{row['EIS 피팅']:.4f} {row['단위']}",
                            "차이":    f"{row['차이']:+.4f} {row['단위']}" if not np.isnan(row["차이"]) else "—",
                            "오차율":  f"{pct:.1f}% {flag}" if not np.isnan(pct) else "—",
                        })
                    st.dataframe(pd.DataFrame(cmp_display), width='stretch', hide_index=True)

                    valid_pcts = [r["오차율"] for r in comparison if not np.isnan(r["오차율"])]
                    if valid_pcts:
                        r_vals = [r["오차율"] for r in comparison if r["파라미터"] in ("Rs","R₁","R₂") and not np.isnan(r["오차율"])]
                        avg_r_err = np.mean(r_vals) if r_vals else np.mean(valid_pcts)
                        if avg_r_err < 5:
                            st.success(f"✅ 저항 파라미터 평균 오차 {avg_r_err:.1f}% — DCIM이 EIS를 잘 모사합니다.")
                        elif avg_r_err < 15:
                            st.warning(f"⚠️ 저항 파라미터 평균 오차 {avg_r_err:.1f}% — 측정 조건 차이를 확인하세요.")
                        else:
                            st.error(f"❌ 저항 파라미터 평균 오차 {avg_r_err:.1f}% — C-rate 또는 피팅 창 재검토가 필요합니다.")
        else:
            st.info("위의 **🔬 EIS 피팅 실행** 버튼을 클릭하면 결과가 표시됩니다.")


# ──────────────────────────────────────────────
# Tab 5: Diagnostics
# ──────────────────────────────────────────────

def render_tab_diag() -> None:
    st.subheader("🏥 DCIM 기반 셀 진단")
    st.caption(
        "SOH 추적 · 함침불량 감지 · 자가방전 감지 기능을 제공합니다. "
        "DCIM 분석 결과(Rs, R1, C1, R2, C2)를 직접 입력하거나, 분석 탭에서 자동으로 가져올 수 있습니다."
    )

    st.markdown("### 🔢 파라미터 입력")
    diag_mode = st.radio(
        "입력 방식",
        ["분석 탭 결과 자동 가져오기", "직접 입력"],
        horizontal=True,
    )

    if diag_mode == "분석 탭 결과 자동 가져오기":
        # Bug fix: 올바른 세션 키("fit_result")와 FitResult 속성 접근
        # 원래 코드는 존재하지 않는 "result" 키 + dict.get() API를 사용해 항상 실패
        fit_res = st.session_state.get("fit_result")
        if fit_res is not None:
            d_Rs = fit_res.Rs * 1000.0   # Ω → mΩ (UI 표시용)
            d_R1 = fit_res.R1 * 1000.0
            d_C1 = fit_res.C1
            d_R2 = fit_res.R2 * 1000.0
            d_C2 = fit_res.C2
            st.success(f"분석 결과 로드됨 — Rs={d_Rs:.3f}, R1={d_R1:.3f}, R2={d_R2:.3f} mΩ")
        else:
            st.warning("▶ 먼저 'Fit Result' 탭에서 DCIM 분석을 실행하세요.")
            d_Rs, d_R1, d_C1, d_R2, d_C2 = 2.0, 1.0, 30.0, 3.0, 200.0
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        d_Rs = col1.number_input("Rs (mΩ)", min_value=0.01, max_value=100.0, value=2.0, step=0.01)
        d_R1 = col2.number_input("R1 SEI (mΩ)", min_value=0.01, max_value=50.0, value=1.0, step=0.01)
        d_C1 = col3.number_input("C1 (F)", min_value=0.1, max_value=500.0, value=30.0, step=0.1)
        d_R2 = col4.number_input("R2 Rct (mΩ)", min_value=0.01, max_value=100.0, value=3.0, step=0.01)
        d_C2 = col5.number_input("C2 (F)", min_value=1.0, max_value=5000.0, value=200.0, step=1.0)

    col_a, col_b, col_c, col_d = st.columns(4)
    d_cycle = col_a.number_input("사이클 수", min_value=0, max_value=5000, value=0, step=1)
    d_soc   = col_b.slider("SOC", 0.0, 1.0, 0.6, 0.05)
    d_temp  = col_c.number_input("온도 (°C)", min_value=-40.0, max_value=80.0, value=25.0, step=1.0)
    d_vdrop = col_d.number_input("휴지 V 강하 (mV, 선택)", min_value=0.0, max_value=500.0,
                                  value=0.0, step=1.0,
                                  help="셀을 장시간 휴지시킨 후 관찰된 전압 강하. 0 = 미측정")

    if st.button("🔍 진단 실행", width='stretch', type="primary"):
        # DCIMParams는 이제 Ω 단위 — UI 입력값(mΩ)을 /1000 변환
        params = DCIMParams(
            Rs=d_Rs / 1000.0,
            R1=d_R1 / 1000.0,
            C1=d_C1,
            R2=d_R2 / 1000.0,
            C2=d_C2,
            cycle=d_cycle, soc=d_soc, temp_c=d_temp,
        )
        v_drop_arg = d_vdrop if d_vdrop > 0 else None

        if "cell_diag" not in st.session_state:
            st.session_state.cell_diag = CellDiagnostics()
        diag_obj = st.session_state.cell_diag

        result_d2 = diag_obj.check_all(params, v_drop_mv=v_drop_arg)
        st.session_state["last_diag"] = (result_d2, params, diag_obj)

    if "last_diag" in st.session_state:
        result_d2, params, diag_obj = st.session_state["last_diag"]

        status_color = {
            "정상": "🟢", "주의": "🟡", "경고": "🟠", "위험": "🔴"
        }.get(result_d2.status, "⚪")
        st.markdown(f"## {status_color} 종합 상태: **{result_d2.status}**   &nbsp;|&nbsp;  SOH 추정: **{result_d2.soh_pct:.1f}%**")

        if result_d2.flags:
            for flag in result_d2.flags:
                st.warning(f"⚠️ {flag}")
        else:
            st.success("이상 징후 없음 — 정상 범위입니다.")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📈 SOH 추적")
            det = result_d2.details
            # R_total은 이제 Ω 단위 → mΩ로 변환하여 표시
            r_total_mohm = det["R_total"] * 1000
            st.metric("SOH", f"{result_d2.soh_pct:.1f}%",
                       delta=f"R_total={r_total_mohm:.3f} mΩ")
            st.metric("경보 등급", det["alert"])
            slope = det.get("trend_slope_ohm_per_cycle")
            if slope is not None:
                # Ω/사이클 → μΩ/사이클 표시
                st.metric("열화 속도", f"{slope * 1e6:.2f} μΩ/사이클")

            hist = diag_obj.tracker.soh_history()
            if len(hist) > 1:
                df_hist = pd.DataFrame(hist, columns=["사이클", "SOH (%)"])
                st.line_chart(df_hist.set_index("사이클"))

        with col2:
            st.markdown("### 💧 함침불량 진단")
            wetness = result_d2.details["wetness"]
            sev_icon = {"정상": "✅", "주의": "⚠️", "위험": "🚨", "해당없음 (사이클 > 5)": "ℹ️"}
            st.markdown(f"**심각도:** {sev_icon.get(wetness['severity'], '')} {wetness['severity']}")
            # Rs_meas는 Ω → mΩ 표시
            st.write(f"- Rs 측정값: {wetness['Rs_meas']*1000:.3f} mΩ (기준 < 5.0 mΩ)")
            st.write(f"- R1 측정값: {wetness['R1_meas']*1000:.3f} mΩ (기준 < 4.0 mΩ)")
            st.markdown("**판정 근거:**")
            for r in wetness["reasons"]:
                st.info(r)

        with col3:
            st.markdown("### 🔋 자가방전 진단")
            sd = result_d2.details["self_discharge"]
            sev_icon2 = {"정상": "✅", "주의": "⚠️", "경고": "🟠", "위험": "🚨"}
            st.markdown(f"**심각도:** {sev_icon2.get(sd['severity'], '')} {sd['severity']}")
            if sd["R2_change_pct"] is not None:
                st.write(f"- R2 변화율: {sd['R2_change_pct']:.1f}% (기준 < 30%)")
            if sd["v_drop_mv"] is not None:
                st.write(f"- 휴지 V 강하: {sd['v_drop_mv']:.1f} mV")
            st.markdown("**판정 근거:**")
            for r in sd["reasons"]:
                st.info(r)

        st.markdown("---")
        st.markdown("### 🔵 진단 파라미터 Nyquist 플롯")

        try:
            import plotly.graph_objects as go
            _plotly_ok = True
        except ImportError:
            _plotly_ok = False

        if not _plotly_ok:
            st.warning("plotly가 설치되지 않아 Nyquist 플롯을 표시할 수 없습니다. "
                       "`pip install plotly` 후 앱을 재시작하세요.")
        else:
            # nyquist_from_params는 Ω 반환 → *1000 하여 mΩ 표시
            zr, zi = nyquist_from_params(params)
            zr_mohm = zr * 1000
            zi_mohm = zi * 1000

            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(
                x=zr_mohm, y=zi_mohm,
                mode="lines",
                name=f"사이클 {params.cycle}",
                line=dict(color="#2155A3", width=2.5),
            ))
            # 히스토리 오버레이 (최대 5개)
            colors = ["#aaaaaa", "#888888", "#666666", "#444444"]
            for idx, p in enumerate(diag_obj.tracker.history[-5:]):
                if p.cycle != params.cycle:
                    zr2, zi2 = nyquist_from_params(p)
                    fig_diag.add_trace(go.Scatter(
                        x=zr2 * 1000, y=zi2 * 1000,
                        mode="lines",
                        name=f"사이클 {p.cycle}",
                        line=dict(color=colors[idx % len(colors)], width=1.5, dash="dot"),
                        opacity=0.6,
                    ))

            fig_diag.update_layout(
                title="DCIM 재구성 Nyquist 플롯 (사이클 비교)",
                xaxis_title="Z′ (mΩ)",
                yaxis_title="−Z″ (mΩ)",
                yaxis_scaleanchor="x",
                height=400,
                plot_bgcolor="#F8FBFF",
            )
            st.plotly_chart(fig_diag, width='stretch')

        st.markdown("### 📋 측정값 상세")
        df_params_diag = pd.DataFrame([{
            "항목": "Rs (전해질 저항)", "값": f"{params.Rs*1000:.3f} mΩ",
            "시정수": "—", "물리적 의미": "이온 이동 경로 저항"
        }, {
            "항목": "R1 (SEI 저항)", "값": f"{params.R1*1000:.3f} mΩ",
            "시정수": f"τ1 = {params.tau1*1000:.1f} ms", "물리적 의미": "음극 표면 계면층 저항"
        }, {
            "항목": "R2 (Rct 반응 저항)", "값": f"{params.R2*1000:.3f} mΩ",
            "시정수": f"τ2 = {params.tau2:.2f} s", "물리적 의미": "Li⁺ 삽탈 반응 속도"
        }, {
            "항목": "R_total", "값": f"{params.R_total*1000:.3f} mΩ",
            "시정수": "—", "물리적 의미": "SOH 추정 기반 총 저항"
        }])
        st.dataframe(df_params_diag, width='stretch', hide_index=True)

        if st.button("🗑️ 진단 히스토리 초기화", type="secondary"):
            del st.session_state["cell_diag"]
            del st.session_state["last_diag"]
            st.rerun()

    else:
        st.info("▶ 파라미터 입력 후 '진단 실행' 버튼을 누르세요.")
        st.markdown("""
**진단 항목 설명:**
- **SOH 추적**: Rs, R2의 사이클별 변화로 배터리 수명 상태(%) 추정
- **함침불량 감지**: 신규 셀(cycle ≤ 5)에서 Rs > 5mΩ 또는 R1 > 4mΩ → 전해질 미침투 판정
- **자가방전 감지**: R2 급증(30%+) 또는 휴지 중 전압 강하 → 내부 단락 조기 탐지
        """)


# ──────────────────────────────────────────────
# Tab 6: Export
# ──────────────────────────────────────────────

def render_tab_export() -> None:
    st.markdown(
        '<div class="tab-desc">'
        '분석 결과를 Excel 또는 텍스트 파일로 내보냅니다. '
        'Excel 파일은 파라미터, 피팅 데이터, 나이퀴스트 데이터 세 시트로 구성됩니다.'
        '</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.fit_result
    if result is not None:
        col1, col2 = st.columns(2)

        with col1:
            excel_bytes = export_results_excel(
                result,
                st.session_state.t_fit,
                st.session_state.V_fit,
                st.session_state.V_pred,
                (st.session_state.re_z, st.session_state.neg_im_z),
            )
            st.download_button(
                label="📊 Excel 보고서 다운로드 (.xlsx)",
                data=excel_bytes,
                file_name="dcim_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch',
            )
            st.caption("시트 구성: Parameters / Fit Data / Nyquist")

        with col2:
            report_str = export_report_text(result)
            st.download_button(
                label="📄 텍스트 보고서 다운로드 (.txt)",
                data=report_str,
                file_name="dcim_report.txt",
                mime="text/plain",
                width='stretch',
            )
            st.caption("간단한 파라미터 요약 텍스트")

        st.subheader("📋 보고서 미리보기")
        st.code(report_str, language=None)

    else:
        st.info("▶ 분석 실행 후 내보내기가 활성화됩니다.")
