# Overthinking

이미 생성된 추론을 여러 길이로 잘라 모델에 다시 입력하고, 이어서 생성한 답안이 어떻게 달라지는지 비교하는 코드입니다. 추론의 어느 시점에 답을 낼 수 있는지 살펴봅니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `generate.py`, `baseline.py` | 문제에 대한 모델 응답 생성 |
| `overthinking.py` | `<think>` 구간을 추출하고 비율별로 잘라 후속 응답 생성 |
| `eval_early.py` | 조기 종료 답안 평가 |
| `evaluate_gpqa.py` | GPQA 답안 평가 |
| `grader.py`, `utils.py` | 답안 추출, 채점, JSONL 입출력 |
| `analyze_results.py` | 생성 응답 분석 |
| `data/` | 벤치마크별 입력 데이터 처리 |

## 사용

이 디렉터리에서 `overthinking.py`를 실행합니다. `--base_path`의 `overthinking.jsonl`을 읽고, 같은 경로에 `overthinking_results.jsonl`을 기록합니다. 모델 실행에는 vLLM, PyTorch와 Transformers가 필요합니다.

주요 옵션은 다음과 같습니다.

- `--model_name`: 모델 이름 또는 경로
- `--chat_template`: Jinja 채팅 템플릿 경로
- `--base_path`: 입력·출력 JSONL 디렉터리
- `--tensor_parallel_size`: 모델을 분할할 GPU 수
- `--gpu_memory_utilization`: vLLM GPU 메모리 사용 비율
