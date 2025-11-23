"""라즈베리파이 AP 인터페이스에서 패킷을 캡처해
10개의 네트워크 메타데이터(feature)로 요약하는 모듈.

- capture_window()를 통해 duration_sec 동안의 트래픽을 한 번에 요약한다.
- guardian.py 실시간 감지와 capture_stream.py CSV 수집에서 공통 사용.
"""
import math
from collections import Counter
from typing import List

from scapy.all import sniff, IP, TCP, UDP  

from config import CAMERA_IP

# 캡쳐 대상 인터페이스 (라즈베리파이 AP = wlan0)
INTERFACE = "wlan0"

# 학습/로그 저장용 컬럼 순서 (10개)
FEATURE_COLUMNS = [
    "packet_count",
    "avg_size",
    "unique_src",
    "unique_dst",
    "tcp_count",
    "udp_count",
    "syn_count",
    "conn_attempts",
    "bandwidth_kb",
    "entropy",
]


def _sniff_packets(duration_sec: float):
    """
    지정된 시간 동안 INTERFACE에서 패킷을 수집하고 리스트로 반환.

    - CAMERA_IP가 설정되어 있으면 해당 IP 관련 패킷만 필터링.
    호출: capture_window() 내부에서 duration_sec 구간의 원시 패킷을 모을 때 사용.
    """
    packets = []

    def _collect(pkt):
        packets.append(pkt)

    kwargs = dict(
        iface=INTERFACE,
        prn=_collect,
        store=False,
        timeout=duration_sec,
    )
    # IP카메라만 보고 싶으면 BPF 필터 사용
    if CAMERA_IP:
        kwargs["filter"] = f"host {CAMERA_IP}"

    try:
        sniff(**kwargs)
    except Exception as e:
        # 인터페이스 이름이 틀렸거나, 권한 문제가 있을 경우 등을 대비
        print(f"[CAPTURE] sniff error: {e}")
        return []

    return packets


def _compute_features(packets, duration_sec: float) -> List[float]:
    """
    캡처된 패킷 리스트로부터 10개의 네트워크 메타데이터 feature를 계산.
    호출: capture_window() 내부에서 sniff 결과를 feature 벡터로 변환할 때 사용.
    """
    packet_count = len(packets)
    if packet_count == 0:
        # 패킷이 한 개도 없으면 전부 0으로 채운다.
        return [0.0] * len(FEATURE_COLUMNS)

    sizes = []
    src_ips = []
    dst_ips = []
    tcp_pkts = []
    udp_pkts = []
    syn_count = 0
    conn_pairs = set()  # (src_ip, dst_port) 기준으로 SYN 시도 카운트
    total_bytes = 0

    for pkt in packets:
        try:
            size = len(pkt)
        except Exception:
            size = 0
        sizes.append(size)
        total_bytes += size

        if IP in pkt:
            src_ips.append(pkt[IP].src)
            dst_ips.append(pkt[IP].dst)

        if TCP in pkt:
            tcp_pkts.append(pkt)
            try:
                flags = pkt[TCP].flags
                # SYN 이면서 ACK는 없는 경우만 카운트 (새 연결 시도 느낌)
                if flags & 0x02 and not (flags & 0x10):
                    syn_count += 1
                    if IP in pkt:
                        conn_pairs.add((pkt[IP].src, pkt[TCP].dport))
            except Exception:
                pass
        elif UDP in pkt:
            udp_pkts.append(pkt)

    avg_size = float(sum(sizes) / len(sizes)) if sizes else 0.0
    unique_src = float(len(set(src_ips)))
    unique_dst = float(len(set(dst_ips)))
    tcp_count = float(len(tcp_pkts))
    udp_count = float(len(udp_pkts))
    conn_attempts = float(len(conn_pairs))
    bandwidth_kb = float(total_bytes) / 1024.0 / max(duration_sec, 1e-6)

    
    if src_ips:
        counter = Counter(src_ips)
        total = sum(counter.values())
        probs = [c / total for c in counter.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy = 0.0

    features = [
        float(packet_count),
        float(avg_size),
        float(unique_src),
        float(unique_dst),
        float(tcp_count),
        float(udp_count),
        float(syn_count),
        float(conn_attempts),
        float(bandwidth_kb),
        float(entropy),
    ]
    return features


def capture_window(duration_sec: float = 1.0) -> List[float]:
    """
    duration_sec 동안 패킷을 캡처한 뒤 10개 feature 리스트를 반환하는 상위 헬퍼.

    호출:
        - guardian.py 메인 루프에서 실시간 감지 입력으로 사용.
        - capture_stream.py에서 학습용 CSV 수집을 위해 사용.
    """
    packets = _sniff_packets(duration_sec)
    return _compute_features(packets, duration_sec)
