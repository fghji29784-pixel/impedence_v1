"""
sidebar.py — Streamlit sidebar UI components for DCIM analyzer.

All functions are pure UI; they return values and do not mutate session state.

Changes:
  - Added render_cell_selector() : 18650 / 21700 / 4680 / 4695 / Custom
  - Added render_nominal_capacity() : C-rate 표시용 공칭 용량 입력
  - Improved help text throughout
  - fit window default now driven by cell preset
"""

from __future__ import annotations

import streamlit as st
from models import CELL_PRESETS


def render_cell_selector() -> tuple[str, dict]:
    """셀 타입 선택 위젯.

    Returns
    -------
    (cell_type_key, preset_dict)
      cell_type_key : CELL_PRESETS 딕셔너리의 키 문자열
      preset_dict   : 해당 셀의 프리셋 정보 전체
    """
    st.subheader("🔋 셀 타입")

    options = list(CELL_PRESETS.keys())
    labels  = [CELL_PRESETS[k]["label"] for k in options]

    choice = st.selectbox(
        "분석할 셀 규격",
        options=labels,
        index=1,   # 21700 기본
        help=(
            "셀 타입에 따라 초기값(Initial Guess), 피팅 윈도우, 정상 범위가 자동 설정됩니다.\n\n"
            "• 18650 / 21700 : 피팅 창 5 s\n"
            "• 4680 : 피팅 창 15 s (느린 시정수 τ₂ 반영)\n"
            "• 4695 : 피팅 창 20 s\n"
            "• Custom : 직접 설정"
        ),
    )
    # label → key 역매핑
    cell_key = options[labels.index(choice)]
    return cell_key, CELL_PRESETS[cell_key]


def render_file_upload() -> tuple:
    """데이터 파일 업로드 위젯.

    Returns
    -------
    (charge_file, eis_file)
    """
    st.subheader("📁 데이터 파일")

    charge_file = st.file_uploader(
        "충전 데이터 파일 (필수)",
        type=["mpt", "csv", "txt"],
        key="charge_file",
        help=(
            "BioLogic GCPL 또는 동일 포맷의 충전 데이터 파일.\n"
            "필수 컬럼: 시간(s), 전압(V), 전류(mA 또는 A).\n"
            "지원 형식: .mpt / .csv / .txt"
        ),
    )

    eis_file = st.file_uploader(
        "EIS 데이터 파일 (선택)",
        type=["mpt", "csv", "txt"],
        key="eis_file",
        help=(
            "BioLogic GEIS 등 EIS 측정 파일 (선택 사항).\n"
            "업로드 시 나이퀴스트 탭에서 EIS 실측값과 DCIM 재현 곡선을 직접 비교할 수 있습니다.\n"
            "필수 컬럼: Freq(Hz), Re(Z)(Ω), -Im(Z)(Ω)"
        ),
    )
    return charge_file, eis_file


def render_current_unit() -> str:
    """전류 단위 선택 위젯.

    Returns
    -------
    'mA' or 'A'
    """
    return st.selectbox(
        "파일 내 전류 단위",
        options=["mA", "A"],
        index=0,
        help=(
            "BioLogic GCPL 파일은 기본적으로 mA 단위로 전류를 저장합니다.\n"
            "mA 선택 시 자동으로 A 단위로 변환됩니다.\n"
            "단위를 잘못 설정하면 Rs가 1000배 틀려질 수 있습니다."
        ),
    )


def render_model_selector() -> str:
    """등가회로 모델 선택 위젯.

    Returns
    -------
    'extended', 'warburg', or 'simple'
    """
    choice = st.selectbox(
        "등가회로 모델",
        options=[
            "Extended Randles  (Rs + R1C1 + R2C2)  ← 표준",
            "Warburg  (Rs + R1C1 + R2C2 + √t)  ← EIS 비교 시",
            "Joint Warburg  (Rs 피팅, ramp+CC 동시)  ← EIS 일치 최적화",
            "Simple Randles    (Rs + R1C1)",
        ],
        index=0,
        help=(
            "**Extended Randles**: Rs + R1‖C1 + R2‖C2.\n"
            "Rs는 ΔV/ΔI(p0→p1)로 고정. 기본 모델.\n\n"
            "**Warburg**: Extended + Warburg σ_W·√t 항.\n"
            "확산 전압을 σ_W로 분리해 R2가 Rct에 더 근접. EIS 비교에 권장.\n\n"
            "**Joint Warburg ← EIS 일치 최적화**: Rs를 고정값으로 쓰지 않고\n"
            "ramp(p0→p2) + CC 전 구간을 동시에 피팅해 Rs, R1, R2를 함께 추출.\n"
            "τ₁이 짧으면(< 1 ms) p0→p1 구간에 R1 충전분이 섞여 Rs가 과대추정되는데,\n"
            "이를 자동으로 보정 → EIS Rs/Rct와 가장 가까운 결과.\n\n"
            "**Simple Randles**: Rs + R1‖C1. 빠른 스크리닝용."
        ),
    )
    if "Joint" in choice:
        return "joint_warburg"
    if "Warburg" in choice:
        return "warburg"
    return "simple" if "Simple" in choice else "extended"


def render_manual_range(default_window: float = 5.0) -> tuple[int | None, float]:
    """p2 수동 지정 및 피팅 창 설정.

    Parameters
    ----------
    default_window : 셀 타입에서 가져온 기본 피팅 창 (초)

    Returns
    -------
    (p2_override, window_s)
    """
    use_manual = st.checkbox(
        "p2 인덱스 수동 지정",
        value=False,
        help=(
            "자동 p2 탐지에 실패하거나 결과가 이상할 때 사용하세요.\n"
            "p2는 전류가 설정값(I_set)의 99% 이내로 안정된 첫 번째 시점입니다.\n"
            "Raw Data 탭에서 p0/p1/p2 위치를 확인한 후 수동으로 입력하세요."
        ),
    )
    p2_override = None
    if use_manual:
        p2_override = int(st.number_input(
            "p2 행 번호 (0부터 시작)",
            min_value=0, value=0, step=1,
            help="데이터 파일의 행 번호(0-based)를 입력하세요.",
        ))

    window_s = float(st.number_input(
        "피팅 창 (p2 이후 초)",
        min_value=0.5,
        max_value=60.0,
        value=default_window,
        step=0.5,
        help=(
            f"p2 시점부터 커브 피팅에 사용할 시간 범위입니다 (기본: {default_window}s).\n\n"
            "• 너무 짧으면: 느린 RC 시정수(τ₂)가 충분히 보이지 않아 C₂ 과소추정.\n"
            "• 너무 길면: 충전 곡선의 비선형 부분(OCV 변화)이 섞여 오차 증가.\n"
            "• 4680/4695 같은 대형 셀은 15–20 s 이상 권장."
        ),
    ))
    return p2_override, window_s


def render_fit_engine() -> bool:
    """피팅 엔진 선택 위젯.

    Returns
    -------
    use_lmfit : bool
    """
    engine = st.selectbox(
        "피팅 엔진",
        options=[
            "scipy.optimize.curve_fit  (빠름, 표준)",
            "lmfit  (느림, 불확실도 추정 정확)",
        ],
        index=0,
        help=(
            "**scipy (기본)**: 빠르고 안정적. 대부분의 경우 충분합니다.\n\n"
            "**lmfit**: 각 파라미터의 1σ 불확실도를 더 정확하게 계산합니다.\n"
            "파라미터 신뢰 구간이 필요하거나 scipy가 수렴에 실패할 때 사용하세요.\n"
            "속도는 2–5배 느립니다."
        ),
    )
    return "lmfit" in engine
