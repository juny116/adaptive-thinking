# Adaptive Thinking

언어모델이 추론하는 동안 표현한 confidence를 읽어 추론을 계속할지, 최종 답안을 생성할지 결정하는 코드입니다. 일반 생성, 바로 답하기, 길이 제한, confidence 기반 중단 전략을 같은 실행 흐름에서 비교합니다.

## 구조

```text
adaptive_thinking/
├── core/          # 모델 API, 비동기 실행, 재개 처리, confidence 파싱
├── strategies/    # 문제 하나에 대한 생성 전략
│   └── variants/  # 중단 단어, 토큰 길이, confidence 질문 방식별 변형
├── data/          # 벤치마크 전처리, JSONL 입출력, 추론 구간 분할
├── evaluation/    # 수학·GPQA 답안 추출과 채점
├── analysis/      # 응답 길이, confidence, 설정별 성능 비교
└── apps/          # Streamlit 응답 뷰어
legacy/overthinking/ # 추론을 잘라 후속 답안을 생성하는 도구
```

`core/runner.py`가 입력 JSONL을 읽고 이미 처리한 문제를 건너뛴 뒤, 선택한 전략의 `process_single_example()`을 비동기로 실행합니다. 결과는 문제 인덱스 순서로 저장하고 전략에 맞는 답안 요약을 출력합니다. 생성 전략은 모델에 전달할 메시지와 추론 중단 조건을 담당합니다.

## 설치와 실행

저장소 루트에서 실행합니다.

```bash
pip install -r requirements.txt
python -m adaptive_thinking list
python -m adaptive_thinking run adaptive --config-dir /absolute/path/to/conf --config-name baseline
python -m adaptive_thinking run baseline --config-dir /absolute/path/to/conf --config-name baseline
python -m adaptive_thinking run fixed-length --config-dir /absolute/path/to/conf --config-name baseline
```

생성 명령은 Hydra 설정을 읽습니다. `model`에는 API 연결 정보와 생성 옵션을, `template`에는 `instruction`, `format`, `system`을 지정합니다. `data_path`, `result_path`, `max_examples`가 입력·출력과 처리 범위를 결정합니다. `concurrency`는 동시 처리 수이며 기본값은 20입니다. 루트에 `conf/`가 있으면 기본 설정 경로로 사용합니다.

입력 레코드는 `question`, `answer`, `solution`과 프롬프트 템플릿에서 참조하는 필드를 담습니다. 생성 결과에는 `index`, `messages`, `answer` 등이 기록되며, confidence 전략은 중간 confidence 기록을 추가합니다. 동일한 결과 경로로 다시 실행하면 저장된 `index`를 건너뜁니다.

## 생성 전략

| 명령 (`run` 다음) | 동작 |
|---|---|
| `adaptive` | 추론 중 confidence를 확인하고 중단 여부 결정 |
| `baseline` | 기본 프롬프트로 응답 생성 |
| `direct` | 추론 구간을 닫고 답안 생성 |
| `thinking` | 추론 구간에서 시작하는 기본 생성 |
| `confidence-continue` | confidence 변화에 따라 추가 추론 수행 |
| `deer`, `deer-token-budget` | confidence 임계값과 토큰 예산을 사용하는 생성 |
| `fixed-length`, `fixed-length-completion` | 추론 길이를 제한하는 생성 변형 |
| `alternative-stop`, `wait-stop`, `wait-alternative-stops` | 지정된 전환 단어에서 confidence 확인 |
| `hesitation-stops` | `Wait`, `Alternatively`, `Hmm`에서 추론 흐름 제어 |
| `confidence-prompt-ablation` | confidence 질문의 메시지 배치를 바꾼 변형 |
| `segment-confidence`, `segment-confidence-control` | 분할된 추론에서 자기 평가와 답안 confidence 측정 |

전체 목록은 `python -m adaptive_thinking list`로 볼 수 있습니다. 명령 이름은 하이픈과 밑줄을 모두 받습니다.

## 데이터와 분석

```bash
python -m adaptive_thinking data split-reasoning --input responses.jsonl --output segments.jsonl --mode words --words 300
python -m adaptive_thinking data normalize-answers --input source.jsonl --output normalized.jsonl
python -m adaptive_thinking analyze confidence-statistics --input responses.jsonl --limit 100
python -m adaptive_thinking analyze length-comparison --baseline baseline.jsonl --candidate adaptive.jsonl --tokenizer MODEL_NAME
python -m adaptive_thinking analyze confidence-correlation --input segments-scored.jsonl --output correlation.png
python -m adaptive_thinking evaluate answers --config-dir /absolute/path/to/conf --config-name eval
```

- `split-reasoning`: `</think>` 앞의 추론을 `Wait` 또는 단어 수 기준으로 분할합니다.
- `length-comparison`: 공통 문제 인덱스끼리 응답 토큰 수를 비교하고 평균 감소량의 bootstrap 신뢰구간을 계산합니다.
- `confidence-statistics`: `Wait`·`Alternatively` 빈도와 confidence 평균을 계산합니다.
- `confidence-correlation`: 자기 평가 confidence와 내부 confidence의 상관관계를 그립니다.
- `compare-runs`, `compare-settings`, `compare-files`: 결과 파일을 모아 정확도와 응답 길이를 비교합니다.

분석 그림과 응답 뷰어는 추가 패키지를 설치해 실행합니다.

```bash
pip install -r requirements-analysis.txt
python -m streamlit run adaptive_thinking/apps/viewer.py
```

추론을 비율별로 잘라 답안을 다시 생성하는 도구는 [Overthinking](legacy/overthinking/README.md)에 설명되어 있습니다. 기존 스크립트에서 새 명령으로 옮기는 방법은 [명령 대응표](docs/commands.md)를 참고하세요.

## 개발

```bash
python -m unittest discover -s tests -v
```
