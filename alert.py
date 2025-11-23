"""Guardian 이상 탐지 결과를 Discord 웹훅으로 전송하는 알림 모듈.

- 환경변수 GUARDIAN_WEBHOOK에서 웹훅 URL을 읽어 사용한다.
- guardian.py에서 GUARDIAN_START/HACKING 이벤트 발생 시 호출된다.
"""
import os
from dotenv import load_dotenv
import json
import requests
from datetime import datetime

load_dotenv()

DISCORD_WEBHOOK = os.getenv("GUARDIAN_WEBHOOK")


def send_alert(event_type: str, detail: str, score: float):
    """
    Guardian에서 발생한 이벤트를 Discord embed 메시지로 전송.

    event_type: 알림 제목 또는 이벤트 라벨 (예: '포트 스캔 의심').
    detail: 사람 눈으로 읽을 수 있는 간단한 설명/시간 정보.
    score: AutoEncoder 재구성 오차 등 이상 스코어 (embed 필드로 전달).
    호출: guardian.py에서 모니터링 시작 시, HACKING 진입 시 사용.
    """
    if not DISCORD_WEBHOOK :
        print("[ALERT] Discord webhook not set. Event:", event_type, detail, score)
        return

    content = {
        "username": "Guardian-AI",
        "embeds": [
            {
                "title": f"[{event_type}] 이상 행위 감지",
                "description": detail,
                "color": 0xFF0000,
                "fields": [
                    {"name": "재구성 오차", "value": f"{score:.6f}", "inline": True},
                    {"name": "시간", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True},
                ],
            }
        ],
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK, data=json.dumps(content), headers={"Content-Type": "application/json"})
        print("[ALERT] sent to discord, status:", resp.status_code)
    except Exception as e:
        print("[ALERT] failed:", e)
