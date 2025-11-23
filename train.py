"""단일 AutoEncoder를 학습/재학습하는 스크립트.

- 최초 1시간 idle CSV → train_initial_model_from_csv()
- 이후 정상 구간 CSV → retrain_model_from_csv()

dual 모델(guardian.py)과는 별도로,
간단한 단일 모델 기반 이상 탐지를 위한 학습 파이프라인을 제공한다.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import create_model, INPUT_DIM
from config import (
    AE_MODEL_PATH,
    AE_THRESHOLD_PATH,
)


MODEL_PATH = AE_MODEL_PATH
THRESHOLD_PATH = AE_THRESHOLD_PATH
DEVICE = torch.device("cpu")  # 라즈베리파이 기준


def _train_autoencoder(data: np.ndarray, epochs: int = 30, batch_size: int = 32):
    """
    주어진 feature 배열로 AutoEncoder를 학습하고 threshold를 계산/저장.
    호출: train_initial_model_from_csv(), retrain_model_from_csv() 내부에서 공통 사용.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"data.shape[1]={data.shape[1]} != INPUT_DIM={INPUT_DIM}")

    x = torch.tensor(data, dtype=torch.float32)

    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = create_model().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"[TRAIN] epoch {epoch:03d}/{epochs} loss={epoch_loss:.6f}")

    # threshold 계산
    model.eval()
    with torch.no_grad():
        x = x.to(DEVICE)
        recon = model(x)
        mse_per_sample = ((recon - x) ** 2).mean(dim=1).cpu().numpy()

    mean = float(mse_per_sample.mean())
    std = float(mse_per_sample.std() + 1e-8)
    threshold = mean + 3.0 * std

    print(f"[TRAIN] mean={mean:.6f}, std={std:.6f} → threshold={threshold:.6f}")

    # 저장
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        f.write(f"{threshold:.8f}\n")

    return model, threshold



def train_initial_model_from_csv(csv_path: str):
    """
    1시간 idle 구간 CSV를 읽어 초기 AutoEncoder 모델과 threshold를 학습.
    호출: 이 파일을 직접 실행할 때 (__main__ 블록) 기본 엔트리 포인트로 사용.
    """
    csv_file = Path(csv_path)
    print(f"[TRAIN] initial training from {csv_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    data = np.loadtxt(csv_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    _train_autoencoder(data, epochs=30, batch_size=32)


def retrain_model_from_csv(csv_path: str):
    """
    정상 구간 CSV를 사용해 기존 단일 AutoEncoder 모델을 재학습하고 threshold를 갱신.
    호출: 외부 스케줄러(cron 등)에서 주기적으로 재학습할 때 사용하도록 설계.
    """
    csv_file = Path(csv_path)
    print(f"[TRAIN] retraining from {csv_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    data = np.loadtxt(csv_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    _train_autoencoder(data, epochs=10, batch_size=32)


if __name__ == "__main__":
    # 간단히 테스트하려면:
    #   python train.py data/idle_1h.csv
    import sys

    if len(sys.argv) < 2:
        print("사용법: python train.py <csv_path>")
        raise SystemExit(1)

    train_initial_model_from_csv(sys.argv[1])
