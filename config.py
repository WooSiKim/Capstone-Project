"""Guardian 프로젝트 공통 설정 모듈.

데이터/모델/로그 경로, 학습용 CSV 경로, threshold 파일 경로,
실시간 감지 파라미터(HACK_MIN_ITER 등)를 한 곳에서 관리한다.
"""
from pathlib import Path

# 경로 관련 기본 디렉토리 설정
# 학습/감지 코드에서 공통으로 import해서 사용
ROOT_DIR = Path(__file__).resolve().parent

# 데이터 / 모델 / 로그 디렉토리
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# CSV 경로 (학습용)
IDLE_CSV = DATA_DIR / "idle_1h.csv"       # 1시간 동안의 IDLE 수집 데이터
STREAM_CSV = DATA_DIR / "stream_10m.csv"  # 10분 정도 STREAM 상태 수집 데이터

# AutoEncoder 모델 경로 (dual model)
IDLE_MODEL_PATH = MODELS_DIR / "idle_autoencoder.pth"
STREAM_MODEL_PATH = MODELS_DIR / "stream_autoencoder.pth"

# AutoEncoder 단일 모델 경로 (detect.py에서 사용 가능)
AE_MODEL_PATH = MODELS_DIR / "autoencoder.pt"
AE_THRESHOLD_PATH = MODELS_DIR / "threshold.txt"

# 각 모델별 threshold 저장 파일
IDLE_THRESHOLD_PATH = MODELS_DIR / "idle_threshold.txt"
STREAM_THRESHOLD_PATH = MODELS_DIR / "stream_threshold.txt"

# 학습 관련 하이퍼파라미터 (기본값)
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# 임계값 계산: mean + SIGMA * std
SIGMA = 3.0

# 실시간 감지용: 캡처 윈도우(초)
WINDOW_DURATION = 1.0

# HACK 판정에 필요한 최소 지속 시간 (초 단위)
HACK_MIN_SECONDS = 5.0

# WINDOW_DURATION 기준으로 환산한 반복 횟수
HACK_MIN_ITER = max(1, int(HACK_MIN_SECONDS / WINDOW_DURATION))

# IP 카메라 IP (패킷 필터용)
CAMERA_IP = "192.168.50.22"

# 로그 파일 경로 (guardian 등에서 사용)
LOG_PATH = LOG_DIR / "guardian.log"
