"""
diagnostics.py — DCIM 기반 배터리 셀 진단 모듈
=================================================
SOH 추적 / 함침불량 감지 / 자가방전 감지 기능 제공

사용법:
    from diagnostics import CellDiagnostics
    diag = CellDiagnostics()
    result = diag.check_all(DCIMParams(Rs=0.002, R1=0.001, ..., cycle=50))

Changes from DCIM_claude:
  - DCIMParams.Rs/R1/R2 단위를 mΩ → Ω 으로 통일
    (다른 모든 모듈이 Ω을 사용하므로 맞춤)
  - tau1, tau2 프로퍼티에서 암묵적 /1000 변환 제거
  - Thresholds 상수도 Ω 기준으로 변경 (5.0 mΩ → 0.005 Ω)
  - nyquist_from_params 반환 단위: mΩ → Ω (모델 모듈과 동일)
  - 디스플레이 문자열은 *1000 을 명시적으로 사용해 mΩ 표기 유지
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import warnings


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class DCIMParams:
    """한 번의 DCIM 측정 결과.

    저항 단위: Ω  (FitResult, impedance_2rc 등 다른 모듈과 동일)
    """
    Rs: float       # 전해질 저항 [Ω]
    R1: float       # SEI 저항    [Ω]
    C1: float       # SEI 커패시턴스 [F]
    R2: float       # 전하이동 저항  [Ω]
    C2: float       # 이중층 커패시턴스 [F]
    cycle: int = 0
    soc: float = 0.5      # 0~1
    temp_c: float = 25.0  # 섭씨

    @property
    def tau1(self) -> float:
        """SEI RC 시정수 [s]  τ1 = R1[Ω] × C1[F]"""
        return self.R1 * self.C1

    @property
    def tau2(self) -> float:
        """전하이동 RC 시정수 [s]  τ2 = R2[Ω] × C2[F]"""
        return self.R2 * self.C2

    @property
    def R_total(self) -> float:
        """총 저항 [Ω]  Rs + R1 + R2"""
        return self.Rs + self.R1 + self.R2


@dataclass
class DiagResult:
    """진단 결과 컨테이너"""
    status: str           # "정상" | "주의" | "경고" | "위험"
    soh_pct: float        # 추정 SOH (%)
    flags: List[str]      # 이상 플래그 목록
    details: Dict         # 세부 수치


# ─────────────────────────────────────────────────────────────
# 기준 임계값 (리튬이온 NMC 기준, 단위: Ω)
# ─────────────────────────────────────────────────────────────

class Thresholds:
    # SOH 경보 기준 (R_total 배수)
    SOH_CAUTION_RATIO  = 1.5
    SOH_WARNING_RATIO  = 2.0
    SOH_DANGER_RATIO   = 3.0

    # 함침 불량 기준 (신규 셀) — 단위: Ω
    WETNESS_RS_MAX     = 0.005   # Ω (= 5 mΩ) — 초과 시 함침 불량 의심
    WETNESS_R1_MAX     = 0.004   # Ω (= 4 mΩ) — SEI 미형성 의심
    WETNESS_CYCLE_MAX  = 5       # 사이클 — 신규 셀 판정 기준

    # 자가방전 기준
    SD_R2_DELTA_PCT    = 30.0   # R2 이전 측정값 대비 % 변화
    SD_RS_DELTA_PCT    = 20.0   # Rs 이전 측정값 대비 % 변화

    # 온도 보정 기준 온도
    T_REF              = 25.0   # °C


# ─────────────────────────────────────────────────────────────
# 온도 보정 함수
# ─────────────────────────────────────────────────────────────

def temp_correct_Rs(Rs_meas: float, temp_c: float) -> float:
    """
    측정된 Rs를 25°C 기준값으로 변환.
    전해질 저항은 온도에 선형적으로 반비례.
      Rs_25 = Rs_meas / (1 + 0.022 * (T - 25))
    단위: Ω (입력/출력 모두)
    """
    factor = 1.0 + 0.022 * (temp_c - Thresholds.T_REF)
    return Rs_meas / max(factor, 0.1)


def temp_correct_R2(R2_meas: float, temp_c: float,
                    Ea_eV: float = 0.6) -> float:
    """
    R2(Rct)를 25°C 기준으로 Arrhenius 보정.
      R2_25 = R2_meas * exp(-Ea/k * (1/T - 1/298))
    Ea_eV: 활성화에너지 (기본 0.6 eV, NMC 기준)
    단위: Ω (입력/출력 모두)
    """
    k_eV = 8.617e-5  # 볼츠만 상수 [eV/K]
    T = temp_c + 273.15
    T_ref = 298.15
    factor = np.exp(-Ea_eV / k_eV * (1.0 / T - 1.0 / T_ref))
    return R2_meas * factor


# ─────────────────────────────────────────────────────────────
# SOH 추적
# ─────────────────────────────────────────────────────────────

class SOHTracker:
    """
    사이클마다 DCIM 파라미터를 저장하고 SOH 트렌드를 추적.

    SOH 추정 원리:
        R_total = Rs + R2 (R1 SEI는 포함하되 가중치 낮음)
        SOH(%) = 100 * (1 - (R_total - R0) / (R_EOL - R0))
        여기서 R0 = 초기값, R_EOL = R0×3.0 (EOL 기준)
    """

    def __init__(self, R0_Rs: float = None, R0_R2: float = None):
        """
        R0_Rs, R0_R2: 초기(신규) 셀의 기준값 [Ω].
                      None 이면 첫 번째 측정값을 기준으로 자동 설정.
        """
        self.history: List[DCIMParams] = []
        self.R0_Rs: Optional[float] = R0_Rs
        self.R0_R2: Optional[float] = R0_R2
        self._initialized = (R0_Rs is not None)

    def add(self, params: DCIMParams) -> None:
        """측정 결과 히스토리에 추가"""
        if not self._initialized:
            self.R0_Rs = params.Rs
            self.R0_R2 = params.R2
            self._initialized = True
        self.history.append(params)

    def soh(self, params: Optional[DCIMParams] = None) -> float:
        """
        SOH(%) 추정.
        R_total 증가율 → SOH 감소.
        SOH = 100 - 100*(R_total - R0_total)/(R0_total * 2.0)
        → R_total가 초기의 3배가 되면 SOH=0 (단순 선형 근사)
        """
        if not self._initialized:
            return 100.0
        p = params if params else (self.history[-1] if self.history else None)
        if p is None:
            return 100.0

        # 온도 보정
        Rs_25 = temp_correct_Rs(p.Rs, p.temp_c)
        R2_25 = temp_correct_R2(p.R2, p.temp_c)
        R_total_now = Rs_25 + R2_25

        Rs0_25 = self.R0_Rs
        R2_0_25 = self.R0_R2
        R0_total = Rs0_25 + R2_0_25
        R_eol = R0_total * 3.0   # EOL: 초기의 3배

        soh = 100.0 * (1.0 - (R_total_now - R0_total) / max(R_eol - R0_total, 1e-9))
        return float(np.clip(soh, 0.0, 100.0))

    def soh_history(self) -> List[Tuple[int, float]]:
        """[(cycle, soh%), ...] 반환"""
        return [(p.cycle, self.soh(p)) for p in self.history]

    def alert_status(self, params: DCIMParams) -> str:
        """정상/주의/경고/위험 반환"""
        if not self._initialized:
            return "정상"
        Rs_ratio = params.Rs / max(self.R0_Rs, 1e-9)
        R2_ratio = params.R2 / max(self.R0_R2, 1e-9)
        ratio = max(Rs_ratio, R2_ratio)
        if ratio >= Thresholds.SOH_DANGER_RATIO:
            return "위험"
        elif ratio >= Thresholds.SOH_WARNING_RATIO:
            return "경고"
        elif ratio >= Thresholds.SOH_CAUTION_RATIO:
            return "주의"
        return "정상"

    def trend_slope(self) -> Optional[float]:
        """
        최근 측정값의 R_total 증가 기울기 (Ω/사이클).
        히스토리 < 3이면 None.
        """
        if len(self.history) < 3:
            return None
        cycles  = np.array([p.cycle for p in self.history], dtype=float)
        rtotals = np.array([p.Rs + p.R2 for p in self.history], dtype=float)
        coeffs = np.polyfit(cycles, rtotals, 1)
        return float(coeffs[0])  # Ω/사이클


# ─────────────────────────────────────────────────────────────
# 함침 불량 감지
# ─────────────────────────────────────────────────────────────

def detect_wetness_failure(params: DCIMParams) -> Dict:
    """
    신규 셀(cycle <= WETNESS_CYCLE_MAX)에서 전해질 함침 불량 감지.

    판정 기준:
        1) Rs > 5 mΩ  → 전해질 미침투 (이온 이동 경로 차단)
        2) R1 > 4 mΩ  → SEI 미형성 (Li⁺ 초기 반응 불완전)

    반환:
        {
          "detected": bool,
          "severity": "정상" | "주의" | "위험",
          "reasons": [str, ...],
          "Rs_meas": float,  # Ω
          "R1_meas": float,  # Ω
        }
    """
    reasons = []
    severity = "정상"

    if params.cycle > Thresholds.WETNESS_CYCLE_MAX:
        return {
            "detected": False,
            "severity": "해당없음 (사이클 > 5)",
            "reasons": ["함침불량 판정은 신규 셀(cycle ≤ 5)에만 적용"],
            "Rs_meas": params.Rs,
            "R1_meas": params.R1,
        }

    rs_thresh = Thresholds.WETNESS_RS_MAX
    r1_thresh = Thresholds.WETNESS_R1_MAX

    if params.Rs > rs_thresh:
        reasons.append(
            f"Rs={params.Rs*1000:.2f} mΩ > 기준 {rs_thresh*1000:.0f} mΩ "
            f"→ 전해질 함침 불완전 (이온 이동 경로 부족)"
        )
        severity = "위험"

    if params.R1 > r1_thresh:
        reasons.append(
            f"R1={params.R1*1000:.2f} mΩ > 기준 {r1_thresh*1000:.0f} mΩ "
            f"→ SEI 미형성 또는 불균일 형성"
        )
        severity = "위험" if severity == "위험" else "주의"

    # 동시 발생 시 더 심각
    if params.Rs > rs_thresh * 0.7 and params.R1 > r1_thresh * 0.7:
        if severity == "정상":
            severity = "주의"
        reasons.append("Rs, R1 모두 상승 → 함침 불량 복합 패턴")

    return {
        "detected": len(reasons) > 0,
        "severity": severity,
        "reasons": reasons if reasons else ["이상 없음"],
        "Rs_meas": params.Rs,
        "R1_meas": params.R1,
    }


# ─────────────────────────────────────────────────────────────
# 자가방전 감지
# ─────────────────────────────────────────────────────────────

def detect_self_discharge(
    current: DCIMParams,
    previous: Optional[DCIMParams] = None,
    v_drop_mv: Optional[float] = None
) -> Dict:
    """
    자가방전(self-discharge) 감지.

    판정 기준:
        방법1 — R2 이상: R2가 이전 측정 대비 30%+ 급증 (단기간)
        방법2 — V 강하: 휴지 중 V_drop 이 있으면서 R2 변화 상관

    주의: 온도 변화로도 R2 변동 가능 → 온도 보정 후 비교

    반환:
        {
          "detected": bool,
          "severity": "정상" | "주의" | "경고" | "위험",
          "reasons": [str, ...],
          "R2_change_pct": float or None,
          "v_drop_mv": float or None,
        }
    """
    reasons = []
    severity = "정상"
    r2_change_pct = None

    # 방법1: R2 급변 감지
    if previous is not None:
        R2_prev_25 = temp_correct_R2(previous.R2, previous.temp_c)
        R2_curr_25 = temp_correct_R2(current.R2, current.temp_c)
        Rs_prev_25 = temp_correct_Rs(previous.Rs, previous.temp_c)
        Rs_curr_25 = temp_correct_Rs(current.Rs, current.temp_c)

        if R2_prev_25 > 0:
            r2_change_pct = (R2_curr_25 - R2_prev_25) / R2_prev_25 * 100.0
            if r2_change_pct > Thresholds.SD_R2_DELTA_PCT:
                reasons.append(
                    f"R2 {r2_change_pct:.1f}% 급증 (기준 {Thresholds.SD_R2_DELTA_PCT}%) "
                    f"→ 내부 단락 또는 자가방전 의심"
                )
                severity = "경고"
            elif r2_change_pct > Thresholds.SD_R2_DELTA_PCT * 0.5:
                reasons.append(f"R2 {r2_change_pct:.1f}% 상승 → 자가방전 초기 징후")
                severity = "주의"

        rs_change_pct = (Rs_curr_25 - Rs_prev_25) / Rs_prev_25 * 100.0 if Rs_prev_25 > 0 else 0
        if rs_change_pct > Thresholds.SD_RS_DELTA_PCT:
            reasons.append(
                f"Rs {rs_change_pct:.1f}% 급증 → 전해질 분해 가속 의심"
            )
            severity = max(severity, "주의", key=lambda x: {"정상":0,"주의":1,"경고":2,"위험":3}[x])

    # 방법2: 전압 강하 + R2 이상 복합
    # R2 > 5 mΩ = 0.005 Ω 기준
    if v_drop_mv is not None:
        if v_drop_mv > 50 and current.R2 > 0.005:
            reasons.append(
                f"휴지 중 V 강하 {v_drop_mv:.1f} mV + R2={current.R2*1000:.2f} mΩ↑ "
                f"→ 자가방전 가능성 높음"
            )
            severity = "경고" if severity != "위험" else "위험"
        elif v_drop_mv > 20:
            reasons.append(f"휴지 중 V 강하 {v_drop_mv:.1f} mV → 모니터링 필요")
            if severity == "정상":
                severity = "주의"

    return {
        "detected": len([r for r in reasons if "이상" not in r]) > 0,
        "severity": severity,
        "reasons": reasons if reasons else ["이상 없음"],
        "R2_change_pct": r2_change_pct,
        "v_drop_mv": v_drop_mv,
    }


# ─────────────────────────────────────────────────────────────
# 통합 진단 클래스
# ─────────────────────────────────────────────────────────────

class CellDiagnostics:
    """
    DCIM 기반 통합 셀 진단 클래스.

    사용 예:
        diag = CellDiagnostics()

        # 사이클마다 DCIM 측정 후 추가 (Rs/R1/R2는 Ω 단위)
        params = DCIMParams(Rs=0.0021, R1=0.0009, C1=30, R2=0.0032, C2=200,
                            cycle=50, soc=0.6, temp_c=25)
        result = diag.check_all(params)
        print(result.status, result.soh_pct)
    """

    def __init__(self,
                 R0_Rs: float = None,
                 R0_R2: float = None):
        self.tracker = SOHTracker(R0_Rs=R0_Rs, R0_R2=R0_R2)
        self._prev_params: Optional[DCIMParams] = None

    def check_all(self,
                  params: DCIMParams,
                  v_drop_mv: Optional[float] = None) -> DiagResult:
        """
        전체 진단 실행.
        params: 이번 측정 결과 (Rs/R1/R2 단위: Ω)
        v_drop_mv: 휴지 중 전압 강하 (mV, 선택)
        """
        flags = []
        details = {}

        # ── 1. SOH 추적 ──
        self.tracker.add(params)
        soh = self.tracker.soh(params)
        alert = self.tracker.alert_status(params)
        slope = self.tracker.trend_slope()
        details["soh_pct"] = round(soh, 1)
        details["R_total"] = params.R_total  # Ω
        details["alert"] = alert
        details["trend_slope_ohm_per_cycle"] = slope  # Ω/사이클 (None if < 3 points)
        if alert in ("경고", "위험"):
            flags.append(
                f"SOH 저하: {alert} "
                f"(Rs={params.Rs*1000:.2f} mΩ, R2={params.R2*1000:.2f} mΩ)"
            )

        # ── 2. 함침 불량 ──
        wetness = detect_wetness_failure(params)
        details["wetness"] = wetness
        if wetness["detected"]:
            flags.append(f"함침불량: {wetness['severity']}")

        # ── 3. 자가방전 ──
        sd = detect_self_discharge(params, self._prev_params, v_drop_mv)
        details["self_discharge"] = sd
        if sd["detected"]:
            flags.append(f"자가방전: {sd['severity']}")

        # ── 종합 상태 판정 ──
        severity_rank = {"정상": 0, "주의": 1, "경고": 2, "위험": 3}
        all_severities = [alert, wetness["severity"], sd["severity"]]
        valid_sev = [s for s in all_severities if s in severity_rank]
        overall = max(valid_sev, key=lambda s: severity_rank[s]) if valid_sev else "정상"

        self._prev_params = params

        return DiagResult(
            status=overall,
            soh_pct=soh,
            flags=flags,
            details=details
        )

    def summary_df(self):
        """
        히스토리 전체를 pandas DataFrame으로 반환 (시각화용).
        저항 컬럼 단위: Ω
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas가 필요합니다: pip install pandas")

        rows = []
        for p in self.tracker.history:
            rows.append({
                "cycle": p.cycle,
                "SOC": p.soc,
                "temp_c": p.temp_c,
                "Rs_ohm": p.Rs,
                "R1_ohm": p.R1,
                "C1_F": p.C1,
                "R2_ohm": p.R2,
                "C2_F": p.C2,
                "tau1_s": p.tau1,
                "tau2_s": p.tau2,
                "R_total_ohm": p.R_total,
                "SOH_pct": self.tracker.soh(p),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Nyquist 플롯 데이터 생성 (진단 결과 시각화용)
# ─────────────────────────────────────────────────────────────

def nyquist_from_params(params: DCIMParams,
                        f_min: float = 0.01,
                        f_max: float = 10000.0,
                        n_pts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    DCIM 파라미터 → Nyquist 플롯 데이터 생성.

    Z(jω) = Rs + R1/(1+jωτ1) + R2/(1+jωτ2)
    τ1 = R1[Ω]*C1[F],  τ2 = R2[Ω]*C2[F]  (단위: s)

    반환: (Z_real [Ω], Z_imag_neg [Ω])
    디스플레이에서 mΩ로 변환하려면 호출 측에서 *1000 할 것.
    """
    Rs   = params.Rs
    R1   = params.R1
    R2   = params.R2
    tau1 = params.tau1   # s
    tau2 = params.tau2   # s

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_pts)
    omega = 2 * np.pi * freqs

    Z1 = R1 / (1 + 1j * omega * tau1)
    Z2 = R2 / (1 + 1j * omega * tau2)
    Z  = Rs + Z1 + Z2

    Z_real     = np.real(Z)
    Z_imag_neg = -np.imag(Z)   # Nyquist 관례: −Z″ 를 Y축에

    return Z_real, Z_imag_neg


def nyquist_from_history(tracker: SOHTracker,
                         **kwargs) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    SOHTracker 히스토리의 모든 측정에 대해 Nyquist 데이터 반환.
    반환: [(Z_real [Ω], Z_imag_neg [Ω], cycle), ...]
    """
    results = []
    for p in tracker.history:
        zr, zi = nyquist_from_params(p, **kwargs)
        results.append((zr, zi, p.cycle))
    return results


# ─────────────────────────────────────────────────────────────
# 간단한 테스트
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DCIM 셀 진단 모듈 테스트")
    print("=" * 60)

    diag = CellDiagnostics()

    # 신규 셀 시뮬레이션 (사이클 0) — 저항 단위: Ω
    p0 = DCIMParams(Rs=0.0018, R1=0.0008, C1=30, R2=0.0029, C2=200,
                    cycle=0, soc=0.6, temp_c=25)
    r0 = diag.check_all(p0)
    print(f"\n[사이클 0] 상태: {r0.status}, SOH: {r0.soh_pct:.1f}%")

    # 중기 열화
    p50 = DCIMParams(Rs=0.0024, R1=0.0012, C1=28, R2=0.0041, C2=180,
                     cycle=50, soc=0.6, temp_c=25)
    r50 = diag.check_all(p50)
    print(f"[사이클 50] 상태: {r50.status}, SOH: {r50.soh_pct:.1f}%")

    # 말기 열화
    p300 = DCIMParams(Rs=0.0042, R1=0.0025, C1=22, R2=0.0098, C2=140,
                      cycle=300, soc=0.6, temp_c=25)
    r300 = diag.check_all(p300)
    print(f"[사이클 300] 상태: {r300.status}, SOH: {r300.soh_pct:.1f}%")
    if r300.flags:
        print(f"  → 플래그: {r300.flags}")

    # 함침 불량 테스트 (Rs=7.2 mΩ = 0.0072 Ω)
    print("\n─── 함침 불량 테스트 ───")
    p_wet = DCIMParams(Rs=0.0072, R1=0.0051, C1=15, R2=0.0030, C2=200,
                       cycle=1, soc=0.5, temp_c=25)
    wet = detect_wetness_failure(p_wet)
    print(f"함침불량 감지: {wet['detected']}, 심각도: {wet['severity']}")
    for reason in wet["reasons"]:
        print(f"  • {reason}")

    # Nyquist 데이터
    print("\n─── Nyquist 데이터 생성 ───")
    zr, zi = nyquist_from_params(p0)
    print(f"Z_real range: {zr.min()*1000:.2f} ~ {zr.max()*1000:.2f} mΩ")
    print(f"−Z_imag max: {zi.max()*1000:.2f} mΩ")
    print("\n✅ 진단 모듈 테스트 완료")
