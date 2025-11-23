"""Guardian 실시간 네트워크 이상 탐지 메인 모듈.

- capture_window로 1초 단위 네트워크 메타데이터를 수집하고
- IDLE/STREAM 두 개의 AutoEncoder + 룰 기반 + 패턴 분석으로 이상 여부를 판단하며
- 상태 머신(GuardianState)으로 NORMAL/SUSPECT/HACKING을 관리하고
- 로그 파일 기록과 Discord 알림 전송을 담당한다.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from alert import send_alert
from capture import FEATURE_COLUMNS, capture_window
from config import (
    HACK_MIN_ITER,
    IDLE_MODEL_PATH,
    IDLE_THRESHOLD_PATH,
    LOG_PATH,
    STREAM_MODEL_PATH,
    STREAM_THRESHOLD_PATH,
    WINDOW_DURATION,
)
from model import INPUT_DIM, create_model


# ─────────────────────────────────────────────────────────────
# 룰 기반 임계값
# ─────────────────────────────────────────────────────────────
PACKET_LIMIT: int = 250           # 초당 패킷 개수
UDP_LIMIT: int = 150              # 초당 UDP 패킷 개수
BANDWIDTH_LIMIT_KB: float = 300.0 # 초당 사용 대역폭 (KB)
SYN_LIMIT: int = 40               # 초당 SYN 패킷 개수
CONN_LIMIT: int = 20              # 초당 새로운 연결 시도 수


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────
def _load_model(path: Path) -> torch.nn.Module:
    """
    저장된 AutoEncoder 가중치를 로드.
    호출: main()에서 IDLE/STREAM 모델을 로드할 때 사용.
    """
    model = create_model()
    if not path.exists():
        print(f"[GUARDIAN] WARNING: model file not found: {path}")
        model.eval()
        return model

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"[GUARDIAN] loaded model from {path}")
    return model


def _load_threshold(path: Path) -> float:
    """
    텍스트 파일에서 MSE 임계값을 읽어 float로 반환.
    호출: main()에서 IDLE/STREAM threshold를 로드할 때 사용.
    """
    if not path.exists():
        print(f"[GUARDIAN] WARNING: threshold file not found: {path}")
        return float("inf")

    try:
        value_str = path.read_text(encoding="utf-8").strip()
        v = float(value_str)
        print(f"[GUARDIAN] loaded threshold from {path}: {v:.6f}")
        return v
    except Exception as e:
        print(f"[GUARDIAN] ERROR: threshold load failed: {e}")
        return float("inf")


def _feats_to_tensor(feats: List[float]) -> torch.Tensor:
    """
    feature 리스트를 (1, INPUT_DIM) 형태의 torch.Tensor로 변환.
    호출: main()에서 AutoEncoder 입력 텐서를 만들 때 사용.
    """
    arr = np.array(feats, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] != INPUT_DIM:
        print(f"[GUARDIAN] WARNING: feature dim mismatch, got {arr.shape[1]}, expected {INPUT_DIM}")
        # 길이가 다르면 잘라내거나 0으로 채움
        if arr.shape[1] > INPUT_DIM:
            arr = arr[:, :INPUT_DIM]
        else:
            pad = np.zeros((1, INPUT_DIM - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
    return torch.from_numpy(arr)


def _mse(model: torch.nn.Module, x: torch.Tensor) -> float:
    """
    AutoEncoder 재구성 오차(MSE)를 계산해 float로 반환.
    호출: main()에서 IDLE/STREAM 모델 각각의 reconstruction error를 구할 때 사용.
    """
    with torch.no_grad():
        recon = model(x)
    loss = torch.mean((x - recon) ** 2).item()
    return float(loss)


def _ensure_log_header(path: Path):
    """
    guardian 로그 CSV가 없으면 헤더 라인을 생성.
    호출: _append_log()에서 로그 기록 전에 1회 보장하기 위해 사용.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "timestamp",
            "state",
            "context",           # IDLE / STREAM
            "idle_mse",
            "idle_thr",
            "stream_mse",
            "stream_thr",
            "rule_suspect",
            "rule_reasons",
        ] + FEATURE_COLUMNS
        writer.writerow(header)


def _append_log(
    state: str,
    context: str,
    idle_mse: float,
    idle_thr: float,
    stream_mse: float,
    stream_thr: float,
    rule_suspect: bool,
    rule_reasons: List[str],
    feats_dict: Dict[str, float],
):
    """
    현재 시각, 상태, MSE/threshold, 룰 기반 결과, feature들을 guardian 로그 CSV에 1줄 추가.
    호출: main()의 감지 루프에서 매 윈도우마다 호출.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _ensure_log_header(LOG_PATH)

    row = [
        ts,
        state,
        context,
        f"{idle_mse:.6f}",
        f"{idle_thr:.6f}",
        f"{stream_mse:.6f}",
        f"{stream_thr:.6f}",
        int(rule_suspect),
        ";".join(rule_reasons) if rule_reasons else "",
    ] + [f"{feats_dict.get(col, 0.0)}" for col in FEATURE_COLUMNS]

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
# ─────────────────────────────────────────────────────────────
# 룰 기반 체크
# ─────────────────────────────────────────────────────────────
def check_rule_suspect(feats: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    단순 임계값 기반(rule-based)으로 트래픽 폭주/이상 여부와 이유 목록을 계산.
    호출: main()의 감지 루프에서 룰 기반 탐지 단계에서 사용.
    """
    reasons: List[str] = []

    pkt = feats.get("packet_count", 0.0)
    udp = feats.get("udp_count", 0.0)
    syn = feats.get("syn_count", 0.0)
    conn = feats.get("conn_attempts", 0.0)
    bw = feats.get("bandwidth_kb", 0.0)

    # 1) 대략적인 폭주 계열
    if pkt > PACKET_LIMIT:
        reasons.append(f"패킷 폭주 (packet_count={pkt:.0f} > {PACKET_LIMIT})")

    if udp > UDP_LIMIT:
        reasons.append(f"UDP 폭주 (udp_count={udp:.0f} > {UDP_LIMIT})")

    if bw > BANDWIDTH_LIMIT_KB:
        reasons.append(
            f"대역폭 이상 사용 (bandwidth_kb={bw:.1f} > {BANDWIDTH_LIMIT_KB})"
        )

    if syn > SYN_LIMIT:
        reasons.append(f"SYN 폭주 (syn_count={syn:.0f} > {SYN_LIMIT})")

    if conn > CONN_LIMIT:
        reasons.append(
            f"다수의 연결 시도 (conn_attempts={conn:.0f} > {CONN_LIMIT})"
        )

    # 2) 미세한 SYN/연결 패턴 (1,2,3번 시나리오용)
    #    전체 패킷/대역폭은 크지 않아도, SYN과 새로운 연결 시도가
    #    짧은 시간 안에 여러 번 반복되면 의심으로 본다.
    #    (idle 상태에서 1,2 시나리오가 여기 걸리도록 의도)
    if syn >= 3 and conn >= 3 and not reasons:
        reasons.append(
            f"SYN/연결 시도 증가 (syn_count={syn:.0f}, conn_attempts={conn:.0f})"
        )

    return (len(reasons) > 0, reasons)


def pattern_suspect_and_desc(feats: Dict[str, float], context: str) -> Tuple[bool, str, List[str]]:
    """
    메타데이터 패턴만 보고 공격 타입(포트스캔/로그인 시도/RTSP 탈취/UDP 플러딩 등)을 분류.
    호출: main()에서 공격 라벨과 설명을 얻어 알림 메시지에 활용.
    """
    packet = float(feats.get("packet_count", 0.0))
    udp = float(feats.get("udp_count", 0.0))
    syn = float(feats.get("syn_count", 0.0))
    conn = float(feats.get("conn_attempts", 0.0))
    bw = float(feats.get("bandwidth_kb", 0.0))

    reasons: List[str] = []

    # UDP 비율
    udp_ratio = (udp / packet) if packet > 0 else 0.0

    # ─────────────────────────────────────────────
    # 4) UDP 플러딩 / 패킷 공격 (scenario 4)
    #   - UDP 패킷이 매우 많고 대부분이 UDP
    #   - IDLE / STREAM 상관 없이 동일하게 본다.
    # ─────────────────────────────────────────────
    if udp >= 100 and udp_ratio >= 0.6:
        return True, "UDP 플러딩 / 패킷 공격 의심", ["짧은 시간에 UDP 패킷이 비정상적으로 많음"]

    # TCP 계열 공격: SYN, conn 둘 다 거의 없으면 분류 안 함
    if syn < 2 and conn < 2:
        return False, "", reasons

    # ─────────────────────────────────────────────
    # 1) 포트 스캔 / 서비스 조사 의심 (scenario 1)
    #   - 여러 포트에 SYN/연결 시도
    #   - conn_attempts 가 크고, UDP 비율은 낮음
    #   - IDLE / STREAM 상관없이 동일하게 포트스캔으로 본다.
    # ─────────────────────────────────────────────
    if conn >= 5 and syn >= conn and udp_ratio < 0.4:
        return True, "포트 스캔 / 서비스 조사 의심", ["여러 포트로 TCP 연결 시도가 반복됨"]

    # ─────────────────────────────────────────────
    # 2) HTTP 로그인 / 관리페이지 접근 시도 의심 (scenario 2)
    #   - 소수의 포트(1~3개)에 반복 접속
    #   - SYN은 몇 번 이상, bw도 어느 정도 이상
    #   - UDP 비율은 낮음 (HTTP 위주)
    #   → IDLE / STREAM 모두 같은 라벨 사용
    # ─────────────────────────────────────────────
    if syn >= 3 and 1 <= conn <= 3 and bw >= 0.3 and udp_ratio < 0.3:
        return True, "HTTP 로그인 / 관리페이지 접근 시도 의심", [
            "같은 포트에 반복적인 TCP 접속과 데이터 전송 패턴"
        ]

    # ─────────────────────────────────────────────
    # 3) RTSP 스트림 탈취 / 가로채기 시도 의심 (scenario 3)
    #   - STREAM 상태
    #   - UDP 트래픽이 존재(영상 스트림) + TCP SYN/연결이 반복
    #   - conn_attempts 는 1~3 정도 (같은 RTSP 포트 반복)
    # ─────────────────────────────────────────────
    if context == "STREAM" and syn >= 3 and conn <= 3 and udp >= 10 and udp_ratio > 0.2:
        return True, "RTSP 스트림 탈취 / 가로채기 시도 의심", [
            "스트림 중 추가 TCP 세션이 반복적으로 열리는 패턴"
        ]

    # ─────────────────────────────────────────────
    # 그 밖의 러프한 분류
    # ─────────────────────────────────────────────
    if context == "STREAM" and (syn >= 2 or conn >= 2):
        return True, "스트림 관련 접근 / 재접속 시도 의심", [
            "스트림 중 장비에 대한 TCP 접속 시도가 여러 번 발생"
        ]

    if syn >= 2 or conn >= 2:
        return True, "장비 접근 / 서비스 탐색 시도 의심", [
            "장비에 대한 TCP 접속 시도가 여러 번 발생"
        ]

    # 위 조건에 안 걸리면 패턴 특정 X
    return False, "", reasons


def decide_context(feats: Dict[str, float]) -> str:
    """
    패킷 수/대역폭/UDP 개수를 기준으로 현재 구간을 IDLE 또는 STREAM으로 추정.
    호출: main() 감지 루프에서 매 윈도우마다 컨텍스트 결정.
    """
    pkt = feats.get("packet_count", 0.0)
    bw = feats.get("bandwidth_kb", 0.0)
    udp = feats.get("udp_count", 0.0)

    # 카메라 스트림이 켜진 경우: 패킷과 대역폭이 자연스럽게 증가함
    if pkt > 80 or bw > 150 or udp > 10:
        return "STREAM"
    return "IDLE"


# ─────────────────────────────────────────────────────────────
# 상태 관리
# ─────────────────────────────────────────────────────────────
@dataclass
class GuardianState:
    """
    연속 의심 횟수를 기반으로 NORMAL/SUSPECT/HACKING 상태를 관리하는 상태 머신.
    사용처: main()에서 매 tick마다 update()를 호출해 현재 상태를 갱신.
    """
    consec_suspect: int = 0   # 연속으로 몇 번 SUSPECT / HACK 느낌이었는지
    last_state: str = "NORMAL"
    last_change_ts: float = time.time()

    def update(self, is_suspect: bool) -> str:
        """
        is_suspect 플래그에 따라 consec_suspect를 증가/감소시키고 상태를 결정.
        호출: main()에서 모델/룰/패턴 결과를 종합한 is_suspect로 매 윈도우마다 호출.
        """
        now = time.time()

        if is_suspect:
            self.consec_suspect += 1
            self.last_change_ts = now
        else:
            # 공격이 멈춘 뒤 3초 지나면 바로 0으로 리셋
            if now - self.last_change_ts > 3.0:
                self.consec_suspect = 0
            elif self.consec_suspect > 0:
                # 3초 내에서는 한 칸씩만 서서히 감소
                self.consec_suspect -= 1

        if self.consec_suspect >= HACK_MIN_ITER:
            self.last_state = "HACKING"
        elif self.consec_suspect > 0:
            self.last_state = "SUSPECT"
        else:
            self.last_state = "NORMAL"

        return self.last_state


# ─────────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────────
def main():
    """
    dual AutoEncoder + rule 기반 Guardian 감지 루프의 엔트리 포인트.

    - IDLE/STREAM 모델 및 threshold 로드
    - capture_window로 feature 수집
    - 모델/룰/패턴 기반 이상 여부 판단
    - GuardianState로 상태 전이 관리
    - 로그 기록 및 HACKING 진입 시 Discord 알림 전송
    호출: 이 파일을 직접 실행할 때 (__main__ 블록) 사용.
    """
    print("============================================")
    print("[GUARDIAN] start dual-AE + rule-based watcher")
    print(f"[GUARDIAN] WINDOW_DURATION = {WINDOW_DURATION}s, HACK_MIN_ITER = {HACK_MIN_ITER}")
    print(
        f"[GUARDIAN] RULE: PACKET_LIMIT={PACKET_LIMIT}, UDP_LIMIT={UDP_LIMIT}, "
        f"BANDWIDTH_LIMIT_KB={BANDWIDTH_LIMIT_KB}, SYN_LIMIT={SYN_LIMIT}, CONN_LIMIT={CONN_LIMIT}"
    )
    print("============================================")

    idle_model = _load_model(IDLE_MODEL_PATH)
    stream_model = _load_model(STREAM_MODEL_PATH)
    idle_thr = _load_threshold(IDLE_THRESHOLD_PATH)
    stream_thr = _load_threshold(STREAM_THRESHOLD_PATH)

    state = GuardianState()
    stream_start_ts = None  # STREAM으로 처음 전환된 시점 (워밍업용)

    # 시작 알림 한 번 보내주고 시작
    try:
        send_alert("GUARDIAN_START", "가디언 모니터링 시작", 0.0)
    except Exception as e:
        print("[GUARDIAN] send_alert start failed:", e)

    while True:
        start_ts = time.time()

        # 1) 윈도우 캡쳐 → feature 벡터
        feats = capture_window(WINDOW_DURATION)
        feats_dict: Dict[str, float] = {name: float(v) for name, v in zip(FEATURE_COLUMNS, feats)}

        # 2) AutoEncoder 재구성 오차 계산
        x = _feats_to_tensor(feats)
        idle_mse = _mse(idle_model, x)
        stream_mse = _mse(stream_model, x)

        idle_anom = idle_mse > idle_thr
        stream_anom = stream_mse > stream_thr

        # 3) 룰 기반 체크
        rule_suspect, rule_reasons = check_rule_suspect(feats_dict)

        # 4) 공격 패턴 기반 탐지 (시나리오 1/2/3/4 구별)
        context = decide_context(feats_dict)
        pattern_suspect, attack_label, pattern_reasons = pattern_suspect_and_desc(
            feats_dict, context
        )

        # ------- STREAM 워밍업 처리 (스트림 켜자마자 이상행위 방지) -------
        now_ts = time.time()
        if context == "STREAM":
            if stream_start_ts is None:
                stream_start_ts = now_ts
            stream_warmup = (now_ts - stream_start_ts) < 3.0
        else:
            stream_start_ts = None
            stream_warmup = False
        # --------------------------------------------------------------------

        # 최종 이상 여부 결정
        if context == "IDLE":
            # IDLE에서는 idle 모델 기반 + rule + 패턴 감지
            model_anom = idle_anom
            is_suspect = model_anom or rule_suspect or pattern_suspect

        else:  # STREAM
            if stream_warmup:
                # 스트림 ON 직후 3초는 AE 무시, rule + pattern만 적용
                model_anom = False
                is_suspect = rule_suspect or pattern_suspect
            else:
                # 안정된 스트림 상태에서는 스트림 모델 사용
                model_anom = stream_anom
                is_suspect = model_anom or rule_suspect or pattern_suspect

        # 상태 업데이트
        cur_state = state.update(is_suspect)

        now_str = datetime.now().strftime('%H:%M:%S')
        # 디버깅 출력
        print(
            f"[{now_str}] "
            f"state={cur_state:8s} "
            f"ctx={context:6s} "
            f"packet={feats_dict.get('packet_count', 0):.0f} "
            f"udp={feats_dict.get('udp_count', 0):.0f} "
            f"syn={feats_dict.get('syn_count', 0):.0f} "
            f"conn={feats_dict.get('conn_attempts', 0):.0f} "
            f"bw={feats_dict.get('bandwidth_kb', 0.0):.1f}KB "
            f"consec={state.consec_suspect}"
        )

        # 6) 로그 파일에 저장
        _append_log(
            cur_state,
            context,
            idle_mse,
            idle_thr,
            stream_mse,
            stream_thr,
            rule_suspect,
            rule_reasons,
            feats_dict,
        )

        # 7) 알림 조건: HACKING 으로 처음 진입했을 때
        if cur_state == "HACKING" and state.consec_suspect == HACK_MIN_ITER:
            # 공격 라벨이 없으면 그냥 "이상 트래픽"으로 표시
            label = attack_label or "이상 트래픽"
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 디스코드에 올라가는 제목
            title = f"[{label}] 이상 행위 감지"

            # 디테일은 최대한 짧게
            detail = "\n".join(
                [
                    f"패턴 분석: {label}",
                    "시간",
                    ts_str,
                ]
            )

            if context == "IDLE":
                alert_mse = idle_mse
            else:
                alert_mse = stream_mse

            try:
                send_alert(title, detail, alert_mse)
            except Exception as e:
                print("[GUARDIAN] send_alert hack failed:", e)

        # 8) 캡쳐 윈도우 시간 보정 (필요하면 약간 쉬기)
        elapsed = time.time() - start_ts
        if elapsed < WINDOW_DURATION * 0.2:
            # sniff 타임아웃이 예상보다 빨리 끝나면 살짝 쉬어 줌
            time.sleep(WINDOW_DURATION * 0.2 - elapsed)


if __name__ == "__main__":
    main()
