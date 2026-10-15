# Adaptive Thinking

언어모델의 추론 길이와 중단 시점을 제어하는 실험 코드입니다. 모델이 표현한 confidence를 읽어 추론을 이어가거나 답안을 생성하고, 일반 생성 및 길이 제한 방식과 비교합니다.

## 주요 기능

- confidence 표현을 파싱하고 추론 계속 여부를 결정하는 생성 루프
- baseline, 고정 길이, 추가 추론 등 생성 방식별 비교
- 수학 문제와 GPQA 답안의 형식 검사 및 채점
- 생성 길이와 정답 여부 분석

## 코드 구조

| 경로 | 역할 |
|---|---|
| `ours.py` | confidence 기반 추론 제어와 답안 생성 |
| `confidence_continue.py` | confidence를 확인하며 추론을 이어가는 생성 흐름 |
| `baseline.py` | 기본 모델 생성과 답안 채점 |
| `fix_len.py`, `fix_len_llm.py` | 길이를 제어하는 생성 변형 |
| `models.py` | 모델 API 호출 |
| `utils.py`, `grader.py` | 답안 추출, 형식 검사, 수학 답안 채점 |
| `gpqa_diamond_utils.py` | GPQA 선택지 답안 처리와 채점 |
| `data/` | 벤치마크별 데이터 전처리 |
| `legacy/overthinking/` | 추론을 여러 지점에서 잘라 후속 답안을 생성하는 실험 |

## 실행 흐름

입력 문제를 읽고 모델에 생성을 요청한 뒤, 생성 방식에 따라 추론을 계속하거나 최종 답안을 받습니다. 답안을 추출해 정답과 비교하고 JSONL로 기록합니다.

`ours.py`, `baseline.py`, `confidence_continue.py`는 Hydra의 `conf/baseline`을 진입점으로 사용합니다. 모델 연결 정보, 생성 옵션, 입력 경로와 프롬프트를 설정합니다. 주요 의존성은 Hydra/OmegaConf, 모델 API 클라이언트와 Transformers입니다.

추론 구간을 직접 잘라 비교하는 코드는 [Overthinking README](legacy/overthinking/README.md)를 참고하세요.
