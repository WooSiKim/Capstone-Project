"""Guardian에서 사용하는 AutoEncoder 모델 정의 모듈.

- 입력 feature 10개를 4차원 잠재공간으로 인코딩하는 단순 MLP AutoEncoder.
- train.py / train_dual_pytorch.py / guardian.py에서 공통으로 사용된다.
"""
import torch
import torch.nn as nn

# 입력 특징 개수: 10개 (캡쳐할 메타데이터 10개 기준)
INPUT_DIM = 10
LATENT_DIM = 4


class AutoEncoder(nn.Module):
    """
    10차원 입력을 4차원 잠재벡터로 압축한 뒤 다시 복원하는 MLP AutoEncoder.
    사용처: create_model()을 통해 학습 스크립트와 guardian 실시간 감지에서 사용.
    """
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def create_model():
    """
    AutoEncoder 인스턴스를 생성해 반환하는 헬퍼 함수.
    호출: 모델 학습(train.py, train_dual_pytorch.py)과 guardian.py에서 공통 사용.
    """
    return AutoEncoder(INPUT_DIM, LATENT_DIM)
