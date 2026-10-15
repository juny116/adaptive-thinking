# Reasoning continuation and early-exit prototypes

추론 생성, baseline, 조기 종료 평가와 답안 채점 유틸리티.

## 주요 코드

- [`overthinking.py`](overthinking.py)
- [`baseline.py`](baseline.py)
- [`generate.py`](generate.py)
- [`eval_early.py`](eval_early.py)

## 보존 범위

GitHub의 초기 상태만 보면 놓칠 수 있는 서버 미커밋 구현을 보존.

소스 18개. 기준 Git HEAD: `9f2d7f33ea774a889ee0a214b4e06ea45a15ac8d` + 당시 디스크 변경분.


## 실행과 검토

이 디렉터리는 코드 검토용 보존본입니다. 데이터·결과·로그·체크포인트·논문 원고·노트북 출력·외부 설정 파일·실행 셸 설정은 포함하지 않았습니다. 원래 상대 경로와 import 구조를 유지했지만, 비공개 설정과 의존성을 복원하기 전에는 실행 가능한 배포본이 아닙니다. API 키는 개인 환경에서 공급해야 하며, `REDACTED_*`와 `/path/to/user`는 제거된 값의 표시입니다.

Python 소스 문법 검사를 수행했습니다. 모델 다운로드, 학습, API 호출, 통합 실행 및 성능 재현은 하지 않았습니다. 소스에 있는 분석 함수는 결과 자체가 아니며 성능 주장을 의미하지 않습니다.

## 출처와 공개 상태

코드와 문서만 공개합니다. 개인 설정과 실험 산출물은 포함하지 않습니다. 기존 저작권 표기와 upstream 출처는 유지하며, 이 모음에 새 라이선스를 일괄 적용하지 않습니다. 외부 코드와 사용자 변경의 권리 확인은 실제 공개 대상별로 진행합니다.
