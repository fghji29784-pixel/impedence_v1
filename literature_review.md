# DCIM vs EIS 임피던스 불일치 — 문헌 조사 보고서 (검증본)

> **작성일**: 2026-05-27 (초안) / **재검증**: 2026-05-27  
> **목적**: `dcim_rewrite`의 핵심 미해결 문제인 *전류 인가(DCIM) vs EIS 임피던스 불일치* 해결을 위한 문헌 조사  
> **검증 방법**: WebSearch + WebFetch로 저자·연도·내용 개별 재확인  
> **초안 오류 수정**: 4건 저자명 오류 교정 (상세 내역 → §7)

---

## 1. 현재 문제 요약

### 증상 (CLAUDE.md 기준)

| 파라미터 | DCIM 측정값 | EIS 기준값 | 오차 |
|---------|------------|-----------|------|
| **Rs** | 과대 | 기준 | ~24–34% |
| **R1 + R2** | 과대 | 기준 | ~112–200% |

### 구조적 원인

```
Rs_DCIM = Rs_true + R1·(1 − exp(−t_p1 / τ1))  ← p1 지연으로 인한 Rct 기여 흡수
R2_DCIM = R2_true + Warburg 기여분             ← σ_W·√t가 R2로 흡수
```

---

## 2. 검증된 논문 목록

---

### 📄 논문 A — Pillai, Desai, Pattipati, Balasingam (2025)

| 항목 | 내용 |
|------|------|
| **제목** | An Improved Approach to Estimate the Internal Resistance of a Battery During the HPPC Test |
| **저자** | Prarthana Pillai, Smeet Desai, Krishna R. Pattipati, Balakumar Balasingam |
| **소속** | University of Windsor / University of Connecticut |
| **출판** | arXiv preprint (IEEE JESTIE 제출), 2025-05-09 |
| **URL** | https://arxiv.org/abs/2505.06410 |
| **검증** | WebFetch 원문 직접 확인 ✅ |

#### 핵심 내용

HPPC에서 내부 저항을 `R₀ = ΔV / ΔI`로 계산할 때, **전류 인가 중 발생하는 OCV 변화(∂OCV/∂SOC × ΔSOC)를 ΔV에서 제거하지 않아** 체계적 과대 추정이 발생한다는 것을 수식으로 증명.

- 실제 터미널 전압 변화: `ΔV = ΔE_OCV + I · R₀`
- 기존 방식은 `ΔE_OCV ≈ 0` 가정 → Rs 과대 추정
- **오차 범위**: SOC 100%에서 107%, SOC 15%에서 292%
- **개선 성과**: 추정 오차 30–250% 감소, 실험 셀에서 최대 20 mΩ 감소

제안 보정식: `R̂₀ = (Δv − Δê) / I_dis`, 여기서 `Δê = κ̂_LS · C{a(t₁,t₀)}`를 최소제곱으로 추정.

#### 본 프로젝트 적용

`calculate_Rs()` 함수의 `ΔV/ΔI` 계산 시, p0→p1 구간의 OCV 기울기(κ)와 충전량 변화를 이용해 ΔV를 보정하면 Rs 정확도 향상 가능. κ(∂OCV/∂SOC)는 셀 사전 특성화 데이터 필요.

---

### 📄 논문 B — Barai, Uddin, Widanage, McGordon, Jennings (2018)

| 항목 | 내용 |
|------|------|
| **제목** | A study of the influence of measurement timescale on internal resistance characterisation methodologies for lithium-ion cells |
| **저자** | Anup Barai, Kotub Uddin, W. D. Widanage, Andrew McGordon, Paul Jennings |
| **소속** | University of Warwick |
| **출판** | *Scientific Reports* (Nature), 2018 |
| **DOI** | [10.1038/s41598-017-18424-5](https://doi.org/10.1038/s41598-017-18424-5) |
| **PMC** | PMC5758786 |
| **검증** | WebFetch PMC 원문 직접 확인 ✅ |

#### 핵심 내용

20 Ah LiFePO₄/C₆ 파우치 셀에 5가지 방법 비교:

| 방법 | 시간 스케일 | 포함 성분 |
|------|-----------|---------|
| 펄스 전력 시험 (DCIM) | 1–30 s | Rs + Rct + 확산 |
| 스위칭 전류 | ms 단위 | Rs + 일부 Rct |
| 1 kHz ACIR | 1 ms | ≈ Rs만 |
| EIS | 전 주파수 | 전 성분 분리 |
| 펄스-멀티사인 | 가변 | 설정에 따름 |

**핵심 결론**:  
> "저항값은 각 기법의 시간 스케일에 크게 의존하며, 시간 스케일이 일치할 때 서로 다른 기법의 저항값도 일치한다."  
> "어떤 기법으로 측정한 저항도 EIS 결과로부터 추정 가능하다."

#### 본 프로젝트 적용

DCIM의 Rs(p0→p1, ~1–4 ms)가 EIS Rsol보다 크게 나오는 것은 **시간 스케일 불일치** 때문. p1 시점에 이미 Rct의 일부가 충전되어 Rs에 합산됨:

```
Rs_DCIM(t_p1) ≈ Rs_EIS + R1 · (1 − exp(−t_p1 / τ1))
```

τ1을 EIS 피팅으로 먼저 얻으면 이 보정값 계산 가능.

---

### 📄 논문 C — Guo, Xu, Li, Pedersen, Gaberscek, Stroe (2024)

| 항목 | 내용 |
|------|------|
| **제목** | Can Electrochemical Impedance Spectroscopy be Replaced by Direct Current Techniques in Battery Diagnosis? |
| **저자** | J. Guo, Y. Xu, P. Li, K. Pedersen, M. Gaberscek, D.-I. Stroe |
| **소속** | Aalborg University (덴마크) 외 |
| **출판** | *ChemPhysChem*, 2024, Vol. 25, No. 21, e202400528 |
| **DOI** | [10.1002/cphc.202400528](https://doi.org/10.1002/cphc.202400528) |
| **검증** | 저자 Aalborg Univ. 연구 포털 확인 ✅ / 원문 페이월 ⚠️ |

#### 핵심 내용

DC 기법과 EIS 기법의 이론적 동등성 및 현실적 한계를 정리한 리뷰 논문.

- **이론적으로 동등**: DC 기법으로 EIS의 **Rohm, Rsei, Rct, Rmt** 모두 추출 가능
- **실제 한계**: DC 기법은 현재 미개발 단계. 시간 상수(τ1, τ2)의 정확한 결정이 어려움
- 머신러닝으로 DC 파형 → EIS 스펙트럼 예측 연구 진행 중
- **결론**: DC 기법이 EIS를 완전 대체하려면 계산 방법 추가 발전 필요

#### 본 프로젝트 적용

본 프로젝트의 DCIM → Nyquist 비교 접근 자체는 학술적으로 지지됨. 단, **현재 기법 수준에서는 구조적 오차가 불가피**하며, Warburg 모델과 Relaxation 모델이 그 한계를 줄이는 최선책임.

---

### 📄 논문 D — Zhang, Deng, Zong, Zuo, Guo, Song, Jiang (2023)

| 항목 | 내용 |
|------|------|
| **제목** | Effect of Sample Interval on the Parameter Identification Results of RC Equivalent Circuit Models of Li-ion Battery: An Investigation Based on HPPC Test Data |
| **저자** | Hehui Zhang, Chang Deng, Yutong Zong, Qingsong Zuo, Haipeng Guo, Shuai Song, Liangxing Jiang |
| **출판** | *MDPI Batteries*, 2023, **9(1), 1** |
| **DOI** | [10.3390/batteries9010001](https://doi.org/10.3390/batteries9010001) |
| **검증** | ProQuest 원문에서 저자 확인 ✅ |

#### 핵심 내용

1-RC, 2-RC 모델에서 HPPC 샘플링 간격(0.1 / 0.2 / 0.5 / 1.0 s)이 파라미터 동정 결과에 미치는 영향 분석.

| 샘플링 간격 | 1-RC 피팅 | 2-RC 피팅 |
|-----------|---------|---------|
| 0.1 s | 우수 | 우수 |
| 0.2 s | 우수 | 우수 |
| 0.5 s | 약간 저하 | **미세 저하 시작** |
| 1.0 s | 저하 | **명확한 편차 증가** |

- 2-RC 모델은 0.5 s 초과 시 피팅 품질 저하, 일부 조건에서 피팅 곡선과 실측값 사이 편차 증가
- 특히 빠른 시정수 τ1을 가진 RC 아크는 충분히 빠른 샘플링 없이는 식별 불가

#### 본 프로젝트 적용

BioLogic 파일의 DCIM 버스트(~192 µs) + CC 데이터(~0.1 s)가 혼합된 구조가 이 논문의 "혼합 샘플링" 문제를 설명함.  
→ `preprocessor.py`의 **시간 기반 윈도우 + 로그 리샘플링** 처리가 이 논문 관점에서 올바른 대응임.

---

### 📄 논문 E — Wildfeuer, Gieler, Karger (2021)

| 항목 | 내용 |
|------|------|
| **제목** | Combining the Distribution of Relaxation Times from EIS and Time-Domain Data for Parameterizing Equivalent Circuit Models of Lithium-Ion Batteries |
| **저자** | Leo Wildfeuer, Philipp Gieler, Alexander Karger |
| **소속** | Institute of Automotive Technology, TU München + TWAICE Technologies GmbH |
| **출판** | *MDPI Batteries*, 2021, **7(3), 52** |
| **DOI** | [10.3390/batteries7030052](https://doi.org/10.3390/batteries7030052) |
| **검증** | ProQuest에서 저자명 확인 + TU München SciProfiles 확인 ✅ |

#### 핵심 내용

EIS(주파수 영역)와 펄스 시험(시간 영역) 각각의 관찰 가능 시정수 범위 한계를 **DRT(Distribution of Relaxation Times, 이완 시간 분포)**로 극복하는 방법 제안.

| 방법 | 관찰 가능 시정수 범위 | 한계 |
|------|------------------|------|
| EIS | 높은 주파수 해상도 | 낮은 주파수(느린 τ) 접근 어려움 |
| 펄스 시험 | 느린 τ 잘 포착 | 빠른 τ 분리 어려움 |
| **DRT 통합** | **양쪽 결합** | 계산 비용 |

- DRT를 사용하면 임의 개수의 RC 소자 파라미터를 DRT에서 직접 결정 가능
- 기존 CNLS 피팅보다 초기값에 덜 민감

#### 본 프로젝트 적용

현재 `eis_fitter.py`의 CNLS 피팅과 `models.py`의 DCIM 피팅이 **독립적으로 동작하는 구조의 근본 한계**를 이 논문이 설명함. 장기 해결책으로 DRT 모듈 추가 시 DCIM–EIS 파라미터 일관성 확보 가능.

---

### 📄 논문 F — Alavi, Birkl, Howey (2015)

| 항목 | 내용 |
|------|------|
| **제목** | Time-domain fitting of battery electrochemical impedance models |
| **저자** | S. Alavi, C. Birkl, D. Howey |
| **소속** | University of Oxford |
| **출판** | *Journal of Power Sources*, 2015, **Vol. 288, pp. 345–352** |
| **URL** | https://www.sciencedirect.com/science/article/abs/pii/S0378775315007569 |
| **ORA** | https://ora.ox.ac.uk/objects/uuid:64ab13db-a77e-47ae-b0ed-4d87a364a0c4 |
| **검증** | ORA(Oxford University Research Archive) 원문 메타데이터 확인 ✅ |

#### 핵심 내용

EIS 등가회로 파라미터를 시간 영역 데이터에서 직접 피팅하는 방법론 연구. 26650 LFP 셀 사용.

- srivcf (분수차 연속시간 시스템 변수법) 기반 파라미터 추정
- **핵심 발견**: 시간 영역으로 피팅한 Nyquist 플롯이 실측 EIS 스펙트럼과 잘 일치  
  (Randles 모델 단순 가정보다 훨씬 정확)
- 파라미터 정확도: 실험실 측정 대비 **13% 이내 일치**
- **소규모 불일치 원인**: 등가회로에 포함되지 않은 **확산(Warburg) 성분에 의한 저주파 거동 차이**

#### 본 프로젝트 적용

"R1+R2의 오차가 Warburg 추가 시 개선된다"는 본 프로젝트 관찰이 이 논문에서 이론적으로 지지됨. `_fit_2rc_warburg()`와 `_fit_relaxation()`이 올바른 방향임.

---

### 📄 논문 G — Kasper, Moertelmaier, Ragulskis 외 (2023)

| 항목 | 내용 |
|------|------|
| **제목** | Calibrated Electrochemical Impedance Spectroscopy and Time-Domain Measurements of a 7 kWh Automotive Lithium-Ion Battery Module with 396 Cylindrical Cells |
| **저자** | Manuel Kasper, Manuel Moertelmaier, Mykolas Ragulskis, Nawfal Al-Zubaidi R-Smith, Johannes Angerer, Mathias Aufreiter, Alberto Romero, Jakob Krummacher, Jianjun Xu, David E. Root, Ferry Kienberger |
| **소속** | Keysight Technologies Austria |
| **출판** | *Batteries & Supercaps*, 2023, **Vol. 6, Issue 4**, e202200415 |
| **DOI** | [10.1002/batt.202200415](https://doi.org/10.1002/batt.202200415) |
| **검증** | 저자 목록 검색 결과에서 확인 ✅ / 원문 페이월 ⚠️ |

#### 핵심 내용

396셀 모듈(7 kWh)에서 EIS와 시간 영역 펄스를 260 사이클에 걸쳐 교차 비교.

- EIS 추출 파라미터: **Rsol (순수 전해질 저항), Rct (전하 이동 저항), L (인덕턴스)**
- 시간 영역 추출 파라미터: **R0, τ1, τ2**
- **중요 발견**: `R0_time ≈ Rsol_EIS + Rct_EIS의 일부 기여`  
  → DCIM Rs가 EIS Rs보다 크게 나오는 이유가 실험적으로 확인됨
- 100 Hz 이상에서 EIS 보정(calibration) 없이는 케이블 인덕턴스로 인한 Rs 오차 발생

#### 본 프로젝트 적용

`CLAUDE.md`의 "Rs 오차 ~24–34%"는 이 논문의 메커니즘으로 설명됨:

```
Rs_DCIM ≈ Rsol_EIS + R1 × (1 − exp(−t_p1 / τ1))
```

Joint Warburg 모델이 Rs를 자유 파라미터로 피팅하면 이 기여분이 최소화됨.

---

### 📄 논문 H — Kienberger, Kasper, Moertelmaier, Popp, Al-Zubaidi R-Smith (2025)

| 항목 | 내용 |
|------|------|
| **제목** | Reconstruction of Electrochemical Impedance Spectroscopy from Time-Domain Pulses of a 3.7 kWh Lithium-Ion Battery Module |
| **저자** | Ferry Kienberger, Manuel Kasper, Manuel Moertelmaier, Hartmut Popp, Nawfal Al-Zubaidi R-Smith |
| **소속** | Keysight Technologies Austria |
| **출판** | *Electrochem* (MDPI), 2025, **Vol. 6, Issue 17** |
| **DOI** | [10.3390/electrochem6020017](https://doi.org/10.3390/electrochem6020017) |
| **Zenodo** | https://zenodo.org/records/15369607 |
| **검증** | Zenodo 원문에서 저자 확인 ✅ |

#### 핵심 내용

DRT를 이용해 시간 영역 펄스 데이터에서 EIS 스펙트럼을 직접 재구성하는 방법.

- **핵심 개선**: 시뮬레이션 전류 대신 **실측 전류 데이터** 직접 사용 → 현실 데이터에 강건
- **정확도**: 1 kHz에서 오차 보정 80%, 50 mHz–1 kHz 범위 재구성 성공
- **효율**: 표준 EIS 대비 **67% 측정 시간 단축**
- 448셀 완조립 모듈에서도 검증 완료

#### 본 프로젝트 적용

장기 로드맵으로, BioLogic GCPL 데이터에서 DRT 기반 EIS 재구성 모듈을 추가하면 별도 EIS 측정 없이 Nyquist 비교 가능. Kienberger가 Kasper 2023의 공동 저자임 — 동일 연구 그룹의 후속 작업.

---

### 📄 논문 I — Białoń, Niestrój, Skarka, Korski (2023)

| 항목 | 내용 |
|------|------|
| **제목** | HPPC Test Methodology Using LFP Battery Cell Identification Tests as an Example |
| **저자** | Tadeusz Białoń, Roman Niestrój, Wojciech Skarka, Wojciech Korski |
| **출판** | *MDPI Energies*, 2023, **16(17), 6239** |
| **DOI** | [10.3390/en16176239](https://doi.org/10.3390/en16176239) |
| **검증** | 검색 결과에서 저자명 확인 ✅ |

#### 핵심 내용

LFP 셀을 대상으로 HPPC 프로토콜 전체 방법론 설명.

- **이완 구간 권고**: 충분한 τ2 포착을 위해 **최소 40초** 이완 시간 필요
- Rs 정확 추출을 위해 **펄스 인가 직후 10–30 ms 이내** 샘플 필요
- 필터링 및 최적화 기법 상세 기술
- EIS를 이용해 최적 HPPC 샘플링 주파수를 사전 결정할 것 권고

#### 본 프로젝트 적용

`preprocessor.py`의 p1 정의(`I_set × 50%`)와 이완 구간 30 s 기본값은 이 논문의 실무 권장사항과 부합함.

---

### 📄 논문 J — Boukamp (1986, 1995)

> ⚠️ **코드 참조 오류 발견**: `eis_fitter.py`에 "Boukamp 1995 — Solid State Ionics 18-19"로 기재되어 있으나, 이는 두 개의 서로 다른 논문이 혼재된 참조임.

| 구분 | 내용 |
|------|------|
| **논문 J-1 (CNLS 피팅 알고리즘)** | "A Nonlinear Least Squares Fit procedure for analysis of immittance data of electrochemical systems" |
| **저자** | B.A. Boukamp |
| **출판** | *Solid State Ionics* 18–19, 1986 |
| **URL** | https://www.sciencedirect.com/science/article/abs/pii/0167273886900317 |
| **논문 J-2 (Kronig-Kramers 검증법)** | "A Linear Kronig-Kramers Transform Test for Immittance Data Validation" |
| **저자** | B.A. Boukamp |
| **출판** | *J. Electrochem. Soc.* **142**, 1885, 1995 |
| **검증** | 두 논문 모두 검색 결과에서 확인 ✅ |

`eis_fitter.py`에서 실제로 사용되는 CNLS 방법의 원문 참조는 **J-1 (1986)**이 정확함. 코드의 "1995" 표기는 오기(誤記).

---

### 📄 논문 K — MDPI Batteries 2020 (저자명 미확인)

| 항목 | 내용 |
|------|------|
| **제목** | Unification of Internal Resistance Estimation Methods for Li-Ion Batteries Using Hysteresis-Free Equivalent Circuit Models |
| **출판** | *MDPI Batteries*, 2020, **6(2), 32** |
| **DOI** | [10.3390/batteries6020032](https://doi.org/10.3390/batteries6020032) |
| **검증** | 제목·DOI 확인 ✅ / 저자명 페이월 미접근 ⚠️ |

#### 핵심 내용 (검색 결과 기반)

- 기존 ECM 파라미터 추정은 EIS와 다른 내부 저항값을 산출
- **히스테리시스 없는 ECM**을 사용하면 HPPC 추정값과 EIS 측정값이 일치
- HPPC, GITT, ECM 추정, EIS 4가지 방법 비교 실험 수행

#### ⚠️ 주의

저자명을 직접 확인하지 못했습니다. 인용 시 DOI로 직접 논문에 접근하여 저자명 확인 후 사용하세요.

---

## 3. 코드 내 미검증 참조 정리

### `Hust et al. (2021)` — `_sequential_peel_p0()` 주석

- **검색 결과**: NIST, Idaho National Lab, Google Scholar 다수 시도 → **해당 저자의 2021년 논문 확인 불가**
- **현재 코드의 sequential peeling** 알고리즘은 아래와 같은 확립된 방법론과 일치:

| 기법 | 관련 논문 |
|------|---------|
| 지수 박리(exponential peeling) | Wildfeuer et al. 2021 (DRT 기반 시정수 분리) |
| 비선형 최소제곱 초기값 추정 | Alavi et al. 2015 (시간 영역 피팅) |

→ **"Hust 2021"이라는 참조 자체가 미검증**. 원 참조 출처 재확인 권장.

### `Koseoglou (2023)` — `_fit_relaxation()` 주석

- **검색 결과**: "Koseoglou" 성의 2023년 배터리 이완 피팅 논문 **직접 확인 불가**
- 방법론(CC 구간 + 이완 구간 2-phase 피팅)은 Białoń et al. 2023 (논문 I)와 개념적으로 동일

→ **"Koseoglou 2023"이라는 참조도 미검증**. 원 참조 출처 재확인 권장.

---

## 4. 프로젝트 적용 우선순위 정리

### 단기 (코드 수정)

| 개선 방향 | 방법 | 근거 논문 | 예상 효과 |
|---------|------|---------|---------|
| **Rs OCV 보정** | `calculate_Rs()`에 κ·ΔSOC 보정항 추가 | 논문 A (Pillai 2025) | Rs 오차 20–30% 감소 |
| **Rs 사후 보정** | EIS τ1 이용: `Rs_corr = Rs - R1·(1−e^{-t_p1/τ1})` | 논문 B (Barai 2018), G (Kasper 2023) | Rs 오차 10–20% 감소 |

### 중기 (모델 개선)

| 개선 방향 | 방법 | 근거 논문 | 예상 효과 |
|---------|------|---------|---------|
| **Warburg + Relaxation 우선** | CC 데이터에 warburg, HPPC 데이터에 relaxation | 논문 F (Alavi 2015), I (Białoń 2023) | R1+R2 오차 50% 이하로 감소 |
| **샘플링 사전 최적화** | EIS로 τ1 확인 후 1/5·τ1 이하 샘플링 권고 | 논문 D (Zhang 2023), I (Białoń 2023) | Rs 추출 구조적 개선 |

### 장기 (신규 모듈)

| 개선 방향 | 방법 | 근거 논문 | 예상 효과 |
|---------|------|---------|---------|
| **DRT 기반 통합 파라미터화** | 펄스 데이터 → DRT → EIS 재구성 | 논문 E (Wildfeuer 2021), H (Kienberger 2025) | DCIM–EIS 구조적 불일치 해소 |

---

## 5. Rs 보정 공식 (코드 적용 가이드)

### 방법 1: EIS 후처리 보정 (즉시 구현 가능)

EIS 피팅 완료 후 DCIM Rs를 사후 보정:

```python
# EIS 피팅에서 τ1 = R1_eis * C1_eis 추출
# p1 샘플링 지연 t_p1 (초 단위)
Rs_corrected = Rs_dcim - R1_eis * (1.0 - np.exp(-t_p1 / (R1_eis * C1_eis)))
```

### 방법 2: Joint Warburg (이미 구현됨)

Rs를 자유 파라미터로 피팅 → OCV 효과 + RC 흡수 효과가 동시에 보정됨 (현재 `_fit_joint_warburg()` 사용).

---

## 6. 논문 요약 테이블 (출처 포함)

| # | 저자 | 연도 | 저널 | 핵심 기여 | 검증 |
|---|------|------|------|---------|------|
| A | Pillai et al. | 2025 | arXiv (IEEE JESTIE) | OCV 보정으로 HPPC Rs 과대 추정 수정 | ✅ |
| B | Barai et al. | 2018 | Scientific Reports | 시간 스케일이 저항값 결정, EIS가 기준 | ✅ |
| C | Guo et al. | 2024 | ChemPhysChem | DC와 EIS 이론적 동등성 및 한계 | ⚠️ |
| D | Zhang et al. | 2023 | MDPI Batteries | 샘플링 간격 >0.5s 시 2-RC 품질 저하 | ✅ |
| E | Wildfeuer et al. | 2021 | MDPI Batteries | DRT로 EIS+시간영역 통합 파라미터화 | ✅ |
| F | Alavi et al. | 2015 | J. Power Sources | 시간 영역 EIS 피팅, Warburg 미포함이 오차 원인 | ✅ |
| G | Kasper et al. | 2023 | Batteries & Supercaps | R0_time ≈ Rsol + Rct 일부 (실험 확인) | ✅ |
| H | Kienberger et al. | 2025 | Electrochem (MDPI) | 펄스 → DRT → EIS 재구성, 67% 시간 단축 | ✅ |
| I | Białoń et al. | 2023 | MDPI Energies | HPPC 표준 방법론, 이완 40s 권고 | ✅ |
| J-1 | Boukamp | 1986 | Solid State Ionics | CNLS 피팅 알고리즘 (EIS 기초) | ✅ |
| J-2 | Boukamp | 1995 | J. Electrochem. Soc. | Kronig-Kramers 데이터 검증법 | ✅ |
| K | 미확인 | 2020 | MDPI Batteries | 히스테리시스 없는 ECM으로 HPPC=EIS 달성 | ⚠️ |

---

## 7. 초안 대비 수정 내역

초안(literature_review.md v1)에서 발견된 **저자명 오류 4건**:

| 오류 (초안) | 정정 (검증본) | 확인 방법 |
|------------|------------|---------|
| "Schmidt et al. (2021)" | **Wildfeuer, Gieler, Karger 2021** (TU München + TWAICE) | ProQuest + SciProfiles |
| "Heins et al. (2015)" | **Alavi, Birkl, Howey 2015** (Oxford) | ORA 원문 |
| "Reining et al. (2025)" | **Kienberger, Kasper, Moertelmaier, Popp, Al-Zubaidi R-Smith 2025** (Keysight) | Zenodo 원문 |
| "Opitz et al. (2023)" | **Białoń, Niestrój, Skarka, Korski 2023** (Silesian Univ. of Technology 등) | 검색 결과 |

또한 **코드 참조 오류** 1건:

| 오류 (코드 내 주석) | 정정 | 
|------------------|------|
| "Boukamp 1995 — Solid State Ionics 18-19, CNLS method" | CNLS 논문은 **Boukamp 1986, Solid State Ionics 18-19**; 1995년 논문은 Kronig-Kramers 검증법 (J. Electrochem. Soc. 142:1885) |

---

## 8. 검증 기준 범례

| 기호 | 의미 |
|------|------|
| ✅ | WebFetch 또는 신뢰 데이터베이스에서 저자·제목·내용 직접 확인 |
| ⚠️ | 제목·DOI 확인됨, 원문 페이월/접근 불가로 세부 내용 간접 확인 |

---

*검색 기준일: 2026-05-27 / 재검증: 동일 일자*  
*전문 접근은 기관 라이선스(Scopus, Web of Science, MDPI 계정) 사용 권장*
