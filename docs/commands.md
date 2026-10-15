# 명령과 모듈 대응표

생성은 `python -m adaptive_thinking run 이름`, 분석은 `python -m adaptive_thinking analyze 이름`, 데이터 처리는 `python -m adaptive_thinking data 이름`으로 실행합니다. Hydra 옵션은 명령 뒤에 그대로 전달합니다.

| 이전 파일 | 현재 모듈 |
|---|---|
| `baseline.py` | `strategies/baseline` |
| `ours.py` | `strategies/adaptive` |
| `confidence_continue.py` | `strategies/confidence_continue` |
| `thinking_baseline.py` | `strategies/thinking` |
| `nothink.py` | `strategies/direct` |
| `deer.py` | `strategies/deer` |
| `deer_max.py` | `strategies/variants/deer_token_budget` |
| `fix_len.py` | `strategies/variants/fixed_length` |
| `fix_len_llm.py` | `strategies/variants/fixed_length_completion` |
| `ablation.py` | `strategies/variants/confidence_prompt_ablation` |
| `only_infinit.py` | `strategies/variants/unlimited_continuation` |
| `hmm.py` | `strategies/variants/hesitation_stops` |
| `wait_only.py` | `strategies/variants/wait_only` |
| `exp4.py` | `strategies/variants/segment_confidence` |
| `exp4_1.py` | `strategies/variants/segment_confidence_control` |
| `alt.py` | `strategies/variants/alternative_stop` |
| `both.py` | `strategies/variants/wait_alternative_stops` |
| `very.py` | `strategies/variants/wait_stop` |
| `highly.py` | `strategies/variants/wait_stop` |
| `eval.py` | `evaluation/score_file` |
| `eval_all.py` | `analysis/compare_runs` |
| `exp4_2.py` | `analysis/compare_settings` |
| `exp4_2_single.py` | `analysis/compare_files` |
| `plot_results.py` | `analysis/plot_benchmarks` |
| `web.py` | `apps/viewer` |
| `prepare_gpqa_diamond.py` | `data/prepare_gpqa` |
| `anal.py` | `analysis/length_comparison` |
| `analyze.py` | `analysis/confidence_statistics` |
| `analyze2.py` | `analysis/confidence_correlation` |
| `change.py` | `data/normalize_answers` |
| `split_data.py` | `data/split_reasoning` |
| `split_len.py` | `data/split_reasoning` |
| `models.py` | `core/model` |
| `utils.py` | `evaluation/answers + data/jsonl + data/preprocessing + core/messages` |
| `grader.py` | `evaluation/grader` |
| `gpqa_diamond_utils.py` | `evaluation/gpqa` |

`split_data.py`의 동작은 `data split-reasoning --mode wait`, `split_len.py`의 동작은 `data split-reasoning --mode words --words 300`에 해당합니다. `very.py`와 `highly.py`는 같은 구현이므로 `run wait-stop`을 사용합니다.

## Overthinking

`legacy/overthinking`에서 실행합니다.

| 이전 파일 | 현재 파일 |
|---|---|
| `analyze_results.py` | `response_statistics.py` |
| `eval.py` | `score_answers.py` |
| `eval_early.py` | `score_truncated_reasoning.py` |
| `overthinking.py` | `truncate_reasoning.py` |
| `generate.py` | `generate_vllm.py` |
| `baseline.py` | `generate_api.py` |
| `models.py` | `model_client.py` |
| `grader.py` | `math_grading.py` |
| `utils.py` | `answer_processing.py` |

`data/strategy.py`는 `data/strategyqa.py`로 바뀌었습니다. 벤치마크 파일명은 소문자로 통일했습니다.
