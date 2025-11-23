"""IDLE / STREAM 두 개의 AutoEncoder를 각각 학습하는 dual 모델 학습 스크립트.

- IDLE_CSV → idle_autoencoder.pth + idle_threshold.txt
- STREAM_CSV → stream_autoencoder.pth + stream_threshold.txt

guardian.py에서 사용하는 실시간 dual-AE 감지 모델을 준비한다.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from model import create_model, INPUT_DIM
from config import (
    IDLE_CSV,
    STREAM_CSV,
    IDLE_MODEL_PATH,
    STREAM_MODEL_PATH,
    IDLE_THRESHOLD_PATH,
    STREAM_THRESHOLD_PATH,
)


DEVICE = torch.device("cpu")  # 라즈베리파이 환경 가정 (GPU 없음)


def _load_csv(path: Path) -> torch.Tensor:
    """
    주어진 CSV 파일을 로드해 (N, INPUT_DIM) 형태의 Tensor로 변환.
    호출: main()에서 IDLE_CSV / STREAM_CSV 각각을 로딩할 때 사용.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    data = np.loadtxt(path, delimiter=",")

    # 1줄만 있는 경우 (10,) → (1, 10) 으로 reshape
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"CSV columns {data.shape[1]} != INPUT_DIM {INPUT_DIM} (file={path})")

    return torch.tensor(data, dtype=torch.float32)


def _train_autoencoder(
    data: torch.Tensor,
    model_path: Path,
    threshold_path: Path,
    epochs: int = 30,
    batch_size: int = 32,
    threshold_scale: float = 3.0,
) -> None:
    """
    단일 데이터셋으로 AutoEncoder를 학습하고 mean+scale*std 기반 threshold를 저장.
    호출: main()에서 IDLE/STREAM 각각에 대해 두 번 호출.
    """
    model = create_model().to(DEVICE)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
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

    # 학습 종료 후 threshold 계산
    model.eval()
    with torch.no_grad():
        data = data.to(DEVICE)
        recon = model(data)
        mse_per_sample = ((recon - data) ** 2).mean(dim=1).cpu().numpy()

    mean = float(mse_per_sample.mean())
    std = float(mse_per_sample.std() + 1e-8)
    threshold = mean + threshold_scale * std

    print(f"[TRAIN] mean={mean:.6f}, std={std:.6f}, scale={threshold_scale} → threshold={threshold:.6f}")

    # 모델 / threshold 저장
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    with threshold_path.open("w") as f:
        f.write(f"{threshold:.8f}\n")


def main():
    print("[TRAIN] dual autoencoder training start")

    # 1) IDLE 모델 학습
    print(f"[TRAIN] loading IDLE csv: {IDLE_CSV}")
    idle_data = _load_csv(IDLE_CSV)
    _train_autoencoder(
        idle_data,
        model_path=IDLE_MODEL_PATH,
        threshold_path=IDLE_THRESHOLD_PATH,
        epochs=30,
        batch_size=32,
        threshold_scale=3.0,   # 기본 3σ
    )

    # 2) STREAM 모델 학습
    print(f"[TRAIN] loading STREAM csv: {STREAM_CSV}")
    stream_data = _load_csv(STREAM_CSV)
    _train_autoencoder(
        stream_data,
        model_path=STREAM_MODEL_PATH,
        threshold_path=STREAM_THRESHOLD_PATH,
        epochs=20,
        batch_size=16,
        threshold_scale=4.0,   # 스트림은 변동성이 더 크다고 보고 약간 완화
    )

    print("[TRAIN] All training completed.")


if __name__ == "__main__":
    main()
