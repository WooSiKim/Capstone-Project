"""IP 카메라가 스트리밍 중일 때의 정상 데이터를 모으는 스크립트.

폰에서 IP카메라 영상을 켜고,
이 스크립트를 실행하면 1초마다 계산된 feature 10개가 
data/stream_10m.csv로 저장된다.
"""

import csv
import time

from capture import capture_window, FEATURE_COLUMNS
from config import STREAM_CSV, WINDOW_DURATION


OUT_CSV = STREAM_CSV  # data/stream_10m.csv


def main():
    OUT_CSV.parent.mkdir(exist_ok=True)
    print("[STREAM CAPTURE] 폰으로 IP카메라 영상을 켜고 그대로 두세요.")
    print(f"[STREAM CAPTURE] 저장 위치: {OUT_CSV}")
    print("중단하려면 Ctrl + C")

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)

        # 기존 코드에서는 헤더 없이 바로 feature만 저장했으므로,
        # 학습 코드와의 호환성을 위해 여기서도 헤더는 쓰지 않는다.
        # 필요하면 아래 주석 해제해서 컬럼명을 첫 줄에 넣을 수 있음.
        # writer.writerow(FEATURE_COLUMNS)

        while True:
            start = time.time()

            feats = capture_window(WINDOW_DURATION)
            writer.writerow(feats)
            f.flush()

            print("[STREAM CAPTURE] features:", feats)

            # 캡처/계산 시간이 WINDOW_DURATION 보다 짧으면 남는 시간만큼 잠시 쉰다.
            elapsed = time.time() - start
            if elapsed < WINDOW_DURATION:
                time.sleep(WINDOW_DURATION - elapsed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STREAM CAPTURE] 종료됨.")
