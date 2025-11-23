# Capstone-Project

라즈베리파이 AP 환경에서 IoT 기기의 네트워크 메타데이터를 실시간 수집·학습하고, AutoEncoder 기반으로 이상 행위를 탐지하는 네트워크 보안 프로젝트입니다.

모델 구조와 코드 작성에는 GitHub, 공식 문서,  ChatGPT의 도움을 받았으며,

전체 파이프라인 설계와 기능 통합, 튜닝, 디버깅은 직접 수행했습니다.



\## 전체 구조



\- \*\*데이터 수집\*\*

&nbsp; - capture.py: 라즈베리파이 AP 인터페이스에서 패킷을 캡처하고 10개 feature로 요약

&nbsp; - capture\_stream.py: 카메라 스트림이 켜진 상태에서 일정 시간 동안 feature를 CSV로 저장



\- \*\*모델 학습\*\*

&nbsp; - model.py: 10차원 입력을 4차원 잠재 공간으로 압축하는 AutoEncoder 정의

&nbsp; - train\_dual\_pytorch.py: IDLE / STREAM 두 종류의 CSV를 읽어

&nbsp;   - idle\_autoencoder.pth / idle\_threshold.txt

&nbsp;   - stream\_autoencoder.pth / stream\_threshold.txt

&nbsp;   를 학습 및 저장



\- \*\*실시간 이상 탐지\*\*

&nbsp; - guardian.py:

&nbsp;   - capture\_window()(capture.py)를 통해 1초 단위 feature 수집

&nbsp;   - dual AutoEncoder + threshold + 룰 기반 + 패턴 분석으로 이상 여부 판단

&nbsp;   - GuardianState로 NORMAL / SUSPECT / HACKING 상태 관리

&nbsp;   - 로그 CSV 기록 및 Discord Webhook 알림 전송



\- \*\*설정 / 공통 모듈\*\*

&nbsp; - config.py: 데이터/모델/로그 경로, threshold 파일 경로, 감지 파라미터 등 공통 설정

&nbsp; - alert.py: Discord Webhook으로 알림 전송

