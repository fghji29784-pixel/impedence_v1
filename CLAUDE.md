# DCIM Battery Analyzer — CLAUDE.md

## 1. 프로젝트 전체 구조

```
DCIM_rewrite/
├── app.py           # Streamlit 진입점 (UI 흐름 전체 조율)
├── sidebar.py       # 사이드바 위젯 함수 모음
├── loader.py        # 파일 로딩 (BioLogic .mpt / CSV / txt)
├── preprocessor.py  # 신호 전처리 (p0/p1/p2 탐지, Rs 계산, 데이터 슬라이싱)
├── models.py        # 등가회로 모델 + 파라미터 피팅 (물리 연산 전부)
├── eis_fitter.py    # EIS CNLS 피팅 (2RC, 2RC_CPE, 3RC, Randles_W)
├── plotter.py       # Matplotlib 그래프 (Nyquist, EIS 피팅, Raw 파형)
├── views.py         # Streamlit 탭별 렌더링 (app.py에서 dispatch)
├── diagnostics.py   # 셀 진단 (SOH 추적, 함침불량, 자가방전)
└── exporter.py      # Excel / 텍스트 결과 내보내기
```

**원격 저장소**: https://github.com/fghji29784-pixel/impedence_v1  
**실행 방법**: `streamlit run app.py`

---

## 2. 각 파일의 역할

### `app.py`
- 페이지 config + 커스텀 CSS (사이드바 그라데이션, 탭 설명 박스, 진단 카드)
- Session state 초기화 (`_STATE_KEYS` 리스트로 일괄 관리)
- 사이드바: 셀 타입 → 파일 업로드 → 분석 설정 → 고급 옵션 → 실행 버튼
- 분석 파이프라인 (▶ 분석 실행 클릭 시):
  1. `load_charge_data` / `load_eis_data`
  2. `detect_I_set` → `find_p0_p1_p2`
  3. `calculate_Rs` (ΔV/ΔI, 2-wire)
  4. `prepare_fit_data` (+ joint_warburg/relaxation 분기별 추가 데이터 준비)
  5. `fit_parameters` → `FitResult`
  6. `compute_nyquist` → re_z, neg_im_z
- 탭 dispatch: 6개 탭을 `views.py`의 함수에 위임

### `sidebar.py`
순수 UI 함수들. 반환값만 있고 session state를 직접 쓰지 않음.
- `render_cell_selector()` → `(cell_key, preset_dict)`
- `render_file_upload()` → `(charge_file, eis_file)`
- `render_current_unit()` → `'mA'` or `'A'`
- `render_model_selector()` → `'extended'|'warburg'|'joint_warburg'|'relaxation'|'simple'`
- `render_manual_range(default_window)` → `(p2_override, window_s, relax_window_s)`
- `render_fit_engine()` → `bool` (use_lmfit)

### `loader.py`
- `load_charge_data(file, current_unit)`: BioLogic GCPL .mpt 파일 파싱. 멀티라인 헤더 자동 스킵, 인코딩 폴백(UTF-8→CP949→Latin-1), mA→A 변환.
- `load_eis_data(file)`: EIS 파일. Freq(Hz), Re(Z)(Ω), -Im(Z)(Ω) 컬럼 자동 탐지.

### `preprocessor.py`
- `detect_I_set(df)`: 전류 데이터에서 설정 전류값 자동 탐지
- `find_p0_p1_p2(df, I_set)`: p0(정지 전), p1(50% I_set, Rs용), p2(99% I_set, 피팅 시작)
- `calculate_Rs(df, idx_p0, idx_p1)`: ΔV/ΔI → 2-wire ohmic Rs
- `prepare_fit_data(df, idx_p2, window_s)`: p2 이후 CC 구간 슬라이싱
- `prepare_joint_fit_data(df, idx_p0, idx_p2, window_s)`: Ramp(p0→p2) + CC 배열 추출 (Joint Warburg용)
- `find_relaxation_start(df, I_set, search_after_idx)`: 전류 차단 지점 탐지 (|I| < 5%·I_set)
- `prepare_relaxation_data(df, idx_relax_start, window_s)`: 이완 구간 t/V 배열 추출

### `models.py`
등가회로 모델 전부 여기에 있음.

**FitResult 데이터클래스**: Rs, R1, C1, R2, C2, σ 불확실도, r², RMSE, sigma_W, tau1/tau2/f1/f2(자동 계산)

**전압 응답 함수**:
- `voltage_response_1rc(t, R1, C1, Vp2, I)` — Simple 모델
- `voltage_response_2rc(t, R1, C1, R2, C2, Vp2, I)` — Extended Randles
- `voltage_response_2rc_warburg(t, R1, C1, R2, C2, sigma_W, Vp2, I)` — 2RC + σ_W·√t
- `voltage_response_relaxation(t, R1, C1, R2, C2, V_relax0, I_pre)` — 전류 차단 후 순수 RC 방전

**임피던스 계산**:
- `impedance_2rc(f, Rs, R1, C1, R2, C2)` — 복소 임피던스 (EIS 재현용)
- `impedance_2rc_warburg(f, Rs, R1, C1, R2, C2, sigma_W)` — Warburg 포함
- `compute_nyquist(Rs, R1, C1, R2, C2, sigma_W=0.0)` — 나이퀴스트 플롯용 (re_z, neg_im_z)

**내부 피팅 함수들**:
- `_fit_1rc`, `_fit_2rc`, `_fit_2rc_warburg`, `_fit_joint_warburg`, `_fit_relaxation`
- `_sequential_peel_p0()`: 로그-선형 peeling으로 R2/τ2 → R1/τ1 순서로 초기값 추정 (Hust 2021)

**진입점**: `fit_parameters(t_fit, V_fit, Rs, I, Vp2, model, ...)` — 모델 키로 dispatch

**CELL_PRESETS**: 18650, 21700, 4680, 4695, Custom — 초기값(p0), 바운드(lb/ub), 피팅 창(fit_window_s)

### `eis_fitter.py`
- **모델**: 2RC, 2RC_CPE, 3RC, Randles_W (Warburg 확산 포함)
- `fit_eis_model(freq, re_z, im_z, model_name, rs_min_bound)`: CNLS (scipy least_squares). Rs 하한 = capacitive arc min × 0.95
- `fit_eis_all_models(freq, re_z, im_z)`: 4개 모델 전부 피팅 → AIC/BIC로 최적 모델 선택
  - capacitive 영역(`-im_z > 0`)만으로 `rs_min_bound` 계산 (inductive tail 제외)
- **EISFitResult**: 파라미터, 불확실도, AIC, BIC, SSR, r²

### `plotter.py`
- `plot_raw_data(df, idx_p0, idx_p1, idx_p2)`: 전압/전류 파형 + p0/p1/p2 마커
- `plot_fit_result(t_fit, V_fit, V_pred)`: 측정값 vs 피팅 곡선
- `plot_nyquist(re_z_dcim, neg_im_z_dcim, df_eis, eis_fit_results, eis_rs_fit)`:
  - DCIM 모델 곡선 + EIS 실측 점 + EIS 피팅 곡선
  - 뷰 범위: capacitive 데이터만으로 계산 (inductive tail이 y축 지배하는 문제 방지)
  - Inductive 점: y_hi×50% 클리핑
- `plot_eis_fit(freq, re_z, neg_im_z, result)`:
  - 피팅 곡선을 `f_max×100` (2 decade extrapolation)까지 연장 → Rs 축 교점 가시화
  - capacitive/inductive 점 분리 표시

### `views.py`
탭별 렌더링. `st.session_state`에서 모든 데이터 읽음. 함수 시그니처에 파라미터 없음.
- `render_tab_raw()`: Raw 파형 + p0/p1/p2 정보, joint_warburg 시 2-wire Rs 표시
- `render_tab_fit()`: 피팅 곡선 + 파라미터 표 + 불확실도. Warburg 시 σ_W 행 추가. joint_warburg 시 2-wire vs fitted Rs 비교
- `render_tab_nyquist()`: Nyquist 그래프 + DCIM vs EIS 파라미터 비교 표. Rs 출처 레이블 표시 ("EIS 피팅" / "capacitive arc 최솟값")
- `render_tab_eis()`: EIS 피팅 결과 (AIC/BIC 모델 선택 + 파라미터 표)
- `render_tab_diag()`: CellDiagnostics 결과 (SOH, 함침, 자가방전)
- `render_tab_export()`: Excel/텍스트 다운로드

### `diagnostics.py`
- `CellDiagnostics.check_all(DCIMParams)`: Rs/R1/R2 이상값 → OK/WARN/ERROR 카드 생성
- SOH 추적, 함침 불량(Rs 급등), 자가방전(OCV drift) 로직
- 파라미터 단위: **Ω** (mΩ 아님) — `*1000` 변환은 표시 문자열에서만 명시적으로 처리

### `exporter.py`
- `export_results_excel(result, t_fit, V_meas, V_pred)`: openpyxl로 xlsx 생성
- `export_results_text(result)`: 텍스트 요약 문자열 반환

---

## 3. 주요 기술 스택

| 패키지 | 용도 |
|--------|------|
| `streamlit` | 웹 UI 프레임워크 |
| `pandas` | 데이터 로딩/처리 |
| `numpy` | 수치 연산 |
| `scipy.optimize.curve_fit` | RC 파라미터 NLS 피팅 |
| `scipy.optimize.least_squares` | EIS CNLS 피팅 |
| `lmfit` | 선택적 피팅 엔진 (불확실도 정밀 추정) |
| `matplotlib` | 그래프 렌더링 |
| `openpyxl` | Excel 내보내기 |

**Python 3.9+** 요구. `from __future__ import annotations` 전 파일 사용.

---

## 4. 완성된 것 / 미완성인 것

### 완성

- **파이프라인 전체 동작**: 파일 로딩 → p0/p1/p2 탐지 → Rs 계산 → 피팅 → Nyquist
- **5개 등가회로 모델**: Extended Randles, Warburg, Joint Warburg, Relaxation, Simple
- **Sequential Peeling 초기값 추정**: Hust(2021) 방법론, `_sequential_peel_p0()`
- **EIS CNLS 피팅 + AIC/BIC 모델 선택**: 4개 모델 자동 비교
- **Nyquist 플롯**: DCIM 곡선 + EIS 실측 + EIS 피팅, 뷰 범위 자동 최적화
- **EIS 피팅 플롯**: 2 decade extrapolation으로 Rs 교점 가시화
- **Rs 일관성**: 그래프와 파라미터 표가 동일한 Rs 출처 사용 (EIS 피팅값 > capacitive arc min 우선순위)
- **셀 타입 프리셋**: 18650/21700/4680/4695/Custom
- **오류 처리**: 파일 인코딩, p2 탐지 실패, Rs 계산 실패 등 사용자 친화적 메시지
- **진단 모듈**: SOH/함침/자가방전 (UI 연동 완료)
- **Excel 내보내기**

### 미완성 / 알려진 한계

- **Rs 오차 ~24–34% (DCIM vs EIS)**: 구조적 문제
  - 2-wire DCIM vs 4-wire EIS Kelvin 측정 — 배선 저항 포함 차이
  - p1 샘플링 지연(0.2~4ms)에서 R1||C1 일부 충전 → Rs 과대 추정
  - Joint Warburg 모델이 Rs를 자유 파라미터로 피팅해 완화하나, 완전 제거 불가
  - 이론적 해결: 4ms 이하 샘플링 + Relaxation 모델 (전류 차단 데이터 필요)

- **R1+R2 오차 ~112–200% (Warburg 사용 시 개선)**: 
  - 2-RC 모델이 Warburg √t 확산 전압을 R2에 흡수 → R2 과대추정
  - Warburg 모델로 σ_W 분리 시 개선되나, 확산 계수 물리적 해석은 별도 검증 필요
  - **Gold standard**: Relaxation 모델 (전류 차단 구간 있는 데이터에서만 동작)

- **Relaxation 모델**: 구현 완료, 단 **전류 차단(HPPC) 데이터** 필요 — 일반 CC 충전 데이터에서는 자동으로 Extended로 폴백

---

## 6. 버그 수정 이력

### [2026-04-06] Relaxation 모델 CC 구간 Warburg 누락 버그 (commit `502bd99`)

**증상**:
- `relaxation` 모델 선택 시 R² ≈ 0.89 (불량), RMSE 0.35 mV
- C2 = 1531 F (물리적으로 비현실적인 값)
- R1+R2 = 10.2 mΩ vs EIS R1+R2 = 1.8 mΩ → **468% 오차**
- 피팅 곡선(빨간 점선)이 실측 데이터(파란 점)보다 전 구간에서 낮음 (잔차 음수)

**근본 원인**:
`_fit_relaxation`의 `model_combined` 내부에서 CC 구간을 `voltage_response_2rc` (순수 2RC)로 피팅했으나,
실제 CC 전압에는 **√t Warburg 확산 성분**이 포함되어 있었음.
Warburg 항이 없으면 옵티마이저가 선형 드리프트를 R2로 흡수 → C2 폭발(τ2 = R2×C2를 맞추기 위해) → R2·C2 부정확.

또한 sequential peeling 호출 시 이완 신호에 음수 부호 오류가 있었음:
```python
# 수정 전 (잘못됨 — V_relax[0] == V_relax0 이므로 항이 상쇄됨)
_sequential_peel_p0(t_relax, -V_relax + V_relax0 + (V_relax[0] - V_relax0), 0.0, -I_set, ...)
# 수정 후
_sequential_peel_p0(t_relax, V_relax0 - V_relax, 0.0, I_set, ...)
```

**수정 내용** (`models.py` — `_fit_relaxation`):
- CC 구간: `voltage_response_2rc` → `voltage_response_2rc_warburg` (σ_W를 5번째 자유 파라미터로 추가)
- 이완 구간: `voltage_response_relaxation` 유지 (순수 RC — τ1/τ2를 깨끗하게 분리)
- Sequential peeling 입력 부호 수정: `V_relax0 - V_relax`, `I_set` 양수로 통일
- `FitResult.sigma_W` 에 피팅된 σ_W 저장

**수정 내용** (`app.py`):
- `V_pred` 계산 분기에 `"relaxation"` 추가 → `voltage_response_2rc_warburg` 사용
  (기존: `else` 브랜치의 `voltage_response_2rc`로 잘못 계산되어 그래프 불일치)

**수정 후 기대 효과**:
- σ_W가 √t 확산 드리프트를 흡수 → R2, C2가 실제 RC 아크 값으로 수렴
- R1+R2 ≈ EIS Rct (Warburg/Joint 모델과 동등한 수준)
- C2가 물리적으로 타당한 범위(수십~수백 F)로 감소

**주의**: Relaxation 모델도 이제 Warburg 계수를 가지므로, `views.py`의 σ_W 행 표시 로직(`getattr(result, "sigma_W", 0.0) > 0.0` 조건)이 자동으로 적용됨.

- **lmfit 경로**: `use_lmfit=True` 분기 코드가 `models.py`에 진입점 존재하나, 내부 피팅 함수들이 아직 lmfit API 미구현. scipy로 fallback 가능성 있음 — 확인 필요.

- **4695 셀 프리셋**: `CELL_PRESETS`에 정의되어 있으나 실제 셀 데이터로 검증 미완료

- **다중 사이클 SOH 추적**: `diagnostics.py`에 로직 있으나 UI에서 히스토리 관리 미구현

---

## 5. 앞으로 작업할 때 주의사항

### Streamlit API
- `use_container_width=True` → **`width='stretch'`** (Streamlit 1.x 신버전 API)
- `use_container_width=False` → **`width='content'`**
- `render_tab_*()` 함수들은 **인수 없음** — session state에서 직접 읽음. 호출 시 인수 넘기지 말 것.
- `render_manual_range()` 반환값은 **3-tuple**: `(p2_override, window_s, relax_window_s)`

### 단위 일관성
- 전류: 모든 내부 처리는 **A** (Ampere). loader.py에서 mA→A 변환.
- 저항: 모든 내부 처리는 **Ω**. 표시할 때만 `×1000`으로 mΩ 변환.
- diagnostics.py: 과거 mΩ → Ω으로 이미 수정됨. 추가 작업 시 mΩ 단위 도입 금지.

### Rs 출처 계층
Nyquist 탭과 파라미터 표에서 Rs(EIS) 표시는 항상 같은 출처 사용:
1. EIS 피팅값 (`eis_fit_results`에서 best model Rs) — 최우선
2. capacitive arc Re(Z) 최솟값 (`re_z_fit.min()`) — EIS 파일 있을 때
3. DCIM 2-wire Rs — EIS 없을 때

이 우선순위 변경 시 `views.py` `render_tab_nyquist()`와 `plotter.py` `plot_nyquist()` 두 곳 동시 수정.

### EIS 피팅 Rs 하한
`eis_fitter.py` `fit_eis_all_models()`에서 capacitive 영역(`-im_z > 0`)만으로 `rs_min_bound` 계산.
inductive tail이 포함되면 Re(Z) min이 실제 Rs보다 낮아져 피팅이 Rs를 과소추정함.

### Nyquist 뷰 범위
`plot_nyquist()`에서 y축 범위는 **capacitive 데이터만** 사용. inductive 점은 `y_hi * 0.5`로 클리핑.
전체 데이터로 뷰 범위 계산하면 inductive tail이 y축을 지배해 아크가 납작해짐.

### EIS 피팅 곡선 생성
`plot_eis_fit()`에서 피팅 곡선은 `param_values`로부터 **새 dense grid** 위에서 재계산.
`r.freq`나 필터링된 `freq_fit` 배열 재사용 금지 — 길이 불일치로 인덱스 오류 발생.

### Joint Warburg 모델
- `app.py`에서 `fit_parameters()` 호출 후 `st.session_state.Rs = result.Rs`로 **피팅된 Rs로 덮어씀**.
- `Rs_dcim_2wire`는 항상 원래 2-wire 추정값 보존.
- `V_ramp`, `I_ramp`, `t_ramp`는 joint_warburg 전용 — 다른 모델에서는 None.

### Sequential Peeling
`_sequential_peel_p0()`은 초기값만 추정. 실제 피팅은 curve_fit이 수행.
peeling이 실패해도 fallback으로 preset p0 사용. `maxfev=20000` 설정으로 수렴 실패 완화.

### 파일 추가 시
새 탭이나 기능 추가: `views.py`에 `render_tab_XXX()` 추가 → `app.py`에서 탭 생성 + dispatch.
session state 키 추가 시 `app.py`의 `_STATE_KEYS` 리스트에 반드시 추가.
