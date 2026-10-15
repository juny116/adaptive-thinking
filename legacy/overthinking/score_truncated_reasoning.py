import json
import argparse
from typing import List, Dict
from tqdm import tqdm
from answer_processing import (
    read_jsonl,
    remove_boxed,
    last_boxed_only_string,
)
from math_grading import AutoScoringJudge
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_correct_samples(
    problem_results: List[Dict],
    max_samples_per_percentage: int = 3,
    max_text_length: int = 3000,
):
    """각 percentage별로 맞은 경우들의 샘플을 출력"""
    print("\n=== 맞은 경우들의 샘플 ===")

    # percentage별로 맞은 문제들 수집
    correct_by_percentage = {}
    for percentage in range(10, 101, 10):
        correct_by_percentage[percentage] = []

        for problem in problem_results:
            if percentage in problem["percentages"]:
                if problem["percentages"][percentage]["correct"]:
                    correct_by_percentage[percentage].append(problem)

    # 각 percentage별로 샘플 출력
    for percentage in sorted(correct_by_percentage.keys()):
        correct_problems = correct_by_percentage[percentage]
        if not correct_problems:
            continue

        print(
            f"\n--- {percentage}%에서 맞은 경우 (총 {len(correct_problems)}개 문제, 최대 {max_samples_per_percentage}개 샘플) ---"
        )

        # 샘플 선택 (앞에서부터 또는 무작위로)
        import random

        sample_problems = random.sample(
            correct_problems, min(max_samples_per_percentage, len(correct_problems))
        )

        for i, problem in enumerate(sample_problems, 1):
            result = problem["percentages"][percentage]
            print(f"\n{i}. Problem {problem['problem_id']}:")
            print(f"   정답: {problem['ground_truth']}")
            print(f"   예측: {result['prediction']}")
            print(f"   생성 길이: {result['generated_length']} tokens")
            print(f"   전체 길이: {result['total_length']} tokens")
            print(f"   Think 길이: {result['think_length']} tokens")

            # 이 문제가 다른 percentage에서도 맞았는지 확인
            correct_percentages = []
            for p in range(10, 101, 10):
                if p in problem["percentages"] and problem["percentages"][p]["correct"]:
                    correct_percentages.append(p)
            print(f"   맞은 percentage들: {correct_percentages}")

            # 실제 생성된 메시지 출력
            generated_text = result.get("generated_text", "")
            if generated_text:
                print(f"   생성된 메시지:")
                print(f"   {'='*50}")
                # 설정된 길이로 제한하여 출력
                if len(generated_text) > max_text_length:
                    print(f"   {generated_text[:max_text_length]}...")
                    print(
                        f"   [메시지가 너무 길어 처음 {max_text_length}자만 표시. 전체 길이: {len(generated_text)}자]"
                    )
                else:
                    print(f"   {generated_text}")
                print(f"   {'='*50}")


def print_problem_analysis(problem_results: List[Dict]):
    """문제별 분석 결과 출력 (100%에서 틀린 문제들만)"""


def evaluate_early_exit_results(data: List[Dict], tokenizer: AutoTokenizer) -> Dict:
    """
    early exit 결과들을 평가하고 성능 지표를 계산합니다.

    Args:
        data: overthinking_results.jsonl에서 읽은 데이터
        tokenizer: 토크나이저

    Returns:
        성능 지표들이 담긴 딕셔너리
    """
    results = {}
    scorer = AutoScoringJudge()
    # 각 percentage별로 성능을 저장할 리스트들
    percentage_results = {i: [] for i in range(10, 101, 10)}

    # 각 문제별 결과를 저장할 리스트
    problem_results = []

    for problem_id, d in enumerate(tqdm(data, desc="Evaluating early exit results")):
        ground_truth = d["answer"]
        problem_data = {
            "problem_id": problem_id,
            "ground_truth": ground_truth,
            "percentages": {},
            "first_correct_percentage": None,
            "always_correct_from": None,
        }

        # 각 early exit 결과에 대해 평가
        for early_exit in d["early_exit"]:
            percentage = early_exit["percentage"]
            generated_text = early_exit["generated_text"]
            truncated_think_length = early_exit.get("truncated_think_length", 0)

            # # 예측 답안 추출
            # try:
            #     prediction = remove_boxed(last_boxed_only_string(generated_text))
            # except Exception as e:
            #     print(f"[ERROR] Problem {problem_id}, percentage {percentage}: {e}")
            #     prediction = ""

            # # 정답 여부 확인
            # is_correct = grade_answer(prediction, ground_truth)
            is_correct, prediction = scorer.judge(ground_truth, generated_text, 1e-2)
            # 생성된 텍스트 길이 계산 (순수 생성 결과만)
            generated_length = len(tokenizer(generated_text)["input_ids"])

            # 전체 응답 길이 (think content + 생성 결과)
            total_length = truncated_think_length + generated_length

            # 문제별 결과에 추가
            problem_data["percentages"][percentage] = {
                "correct": is_correct,
                "prediction": prediction,
                "generated_text": generated_text,  # 실제 생성된 텍스트 추가
                "generated_length": generated_length,  # 순수 생성 결과 길이
                "total_length": total_length,  # 전체 응답 길이
                "think_length": truncated_think_length,  # think content 길이
            }

            # 전체 결과에 추가
            percentage_results[percentage].append(
                {
                    "problem_id": problem_id,
                    "correct": is_correct,
                    "length": generated_length,  # 기존 호환성을 위해 generated_length 사용
                    "generated_length": generated_length,
                    "total_length": total_length,
                    "think_length": truncated_think_length,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                }
            )

        # 첫 번째로 맞춘 percentage 찾기
        for percentage in sorted(problem_data["percentages"].keys()):
            if problem_data["percentages"][percentage]["correct"]:
                problem_data["first_correct_percentage"] = percentage
                break

        # 지속적으로 맞추기 시작하는 percentage 찾기
        consecutive_correct = 0
        for percentage in sorted(problem_data["percentages"].keys()):
            if problem_data["percentages"][percentage]["correct"]:
                consecutive_correct += 1
            else:
                consecutive_correct = 0

            # 3번 연속 맞춘 경우부터 "지속적으로 맞춘다"고 판단
            if consecutive_correct >= 3 and problem_data["always_correct_from"] is None:
                problem_data["always_correct_from"] = (
                    percentage - 20
                )  # 3번 연속이므로 2개 전부터

        problem_results.append(problem_data)

    # 100%에서 맞은 문제들을 제외 (100%에서 틀린 문제들만 분석)
    filtered_problem_results = []
    for problem in problem_results:
        if 100 in problem["percentages"]:
            # 100%에서 틀린 경우만 포함
            if not problem["percentages"][100]["correct"]:
                filtered_problem_results.append(problem)
        else:
            # 100% 데이터가 없는 경우도 포함
            filtered_problem_results.append(problem)

    print(
        f"전체 문제: {len(problem_results)}개, 100%에서 틀린 문제: {len(filtered_problem_results)}개"
    )

    # 필터링된 결과에 맞춰 percentage_results도 업데이트
    filtered_percentage_results = {i: [] for i in range(10, 101, 10)}
    for problem in filtered_problem_results:
        problem_id = problem["problem_id"]
        for percentage in range(10, 101, 10):
            if percentage in problem["percentages"]:
                # 원래 percentage_results에서 해당 문제의 데이터를 찾아서 추가
                for result in percentage_results[percentage]:
                    if result["problem_id"] == problem_id:
                        filtered_percentage_results[percentage].append(result)
                        break

    # 각 percentage별 성능 계산 (필터링된 데이터 사용)
    performance_summary = {}
    for percentage in range(10, 101, 10):
        results_list = filtered_percentage_results[percentage]
        if results_list:
            accuracy = sum(r["correct"] for r in results_list) / len(results_list)
            avg_length = sum(r["length"] for r in results_list) / len(results_list)
            correct_count = sum(r["correct"] for r in results_list)
            total_count = len(results_list)

            performance_summary[percentage] = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
                "avg_length": avg_length,
            }

    return performance_summary, filtered_percentage_results, filtered_problem_results


def print_performance_summary(performance_summary: Dict):
    """성능 요약 출력 (100%에서 틀린 문제들만)"""
    print("\n=== Early Exit Performance Summary (100%에서 틀린 문제들만) ===")
    print(
        f"{'Percentage':<10} {'Accuracy':<10} {'Correct/Total':<15} {'Avg Length':<12}"
    )
    print("-" * 50)

    for percentage in sorted(performance_summary.keys()):
        stats = performance_summary[percentage]
        print(
            f"{percentage}%{'':<7} {stats['accuracy']:.4f}{'':<4} "
            f"{stats['correct_count']}/{stats['total_count']}{'':<6} "
            f"{stats['avg_length']:.1f}"
        )


def save_detailed_results(percentage_results: Dict, output_path: str):
    """상세 결과를 CSV 파일로 저장"""
    detailed_data = []

    for percentage, results_list in percentage_results.items():
        for i, result in enumerate(results_list):
            detailed_data.append(
                {
                    "sample_id": i,
                    "percentage": percentage,
                    "correct": result["correct"],
                    "generated_length": result["generated_length"],
                    "total_length": result["total_length"],
                    "think_length": result["think_length"],
                    "prediction": result["prediction"],
                    "ground_truth": result["ground_truth"],
                }
            )

    df = pd.DataFrame(detailed_data)
    df.to_csv(output_path, index=False)
    print(f"\n상세 결과가 '{output_path}'에 저장되었습니다.")


def plot_performance_curve(performance_summary: Dict, output_path: str):
    """성능 곡선 그래프 생성"""
    percentages = sorted(performance_summary.keys())
    accuracies = [performance_summary[p]["accuracy"] for p in percentages]
    avg_lengths = [performance_summary[p]["avg_length"] for p in percentages]

    # 두 개의 y축을 가진 그래프 생성
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 정확도 그래프
    color = "tab:blue"
    ax1.set_xlabel("Think Content Percentage (%)")
    ax1.set_ylabel("Accuracy", color=color)
    ax1.plot(
        percentages, accuracies, color=color, marker="o", linewidth=2, markersize=6
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 평균 길이 그래프
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Average Length (tokens)", color=color)
    ax2.plot(
        percentages, avg_lengths, color=color, marker="s", linewidth=2, markersize=6
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Early Exit Performance vs Think Content Percentage")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"성능 곡선 그래프가 '{output_path}'에 저장되었습니다.")


def print_problem_analysis(problem_results: List[Dict]):
    """문제별 분석 결과 출력 (100%에서 틀린 문제들만)"""
    print("\n=== Problem-wise Analysis (100%에서 틀린 문제들만) ===")
    print(f"분석 대상: {len(problem_results)}개 문제 (100%에서 맞은 문제 제외)")

    # 첫 번째로 맞춘 percentage 분포
    first_correct_dist = {}
    always_correct_dist = {}
    never_correct_count = 0

    for problem in problem_results:
        if problem["first_correct_percentage"] is not None:
            first_percentage = problem["first_correct_percentage"]
            first_correct_dist[first_percentage] = (
                first_correct_dist.get(first_percentage, 0) + 1
            )
        else:
            never_correct_count += 1

        if problem["always_correct_from"] is not None:
            always_percentage = problem["always_correct_from"]
            always_correct_dist[always_percentage] = (
                always_correct_dist.get(always_percentage, 0) + 1
            )

    print(f"\n첫 번째로 맞춘 percentage 분포:")
    for percentage in sorted(first_correct_dist.keys()):
        print(f"  {percentage}%: {first_correct_dist[percentage]}문제")
    print(f"  Never correct: {never_correct_count}문제")

    print(f"\n지속적으로 맞추기 시작하는 percentage 분포:")
    for percentage in sorted(always_correct_dist.keys()):
        print(f"  {percentage}%: {always_correct_dist[percentage]}문제")

    # 맞은 경우 샘플링 (percentage별로)
    print_correct_samples(problem_results)

    # 문제별 상세 정보 (처음 몇 개만)
    print(f"\n=== 문제별 상세 정보 (처음 10개) ===")
    for i, problem in enumerate(problem_results[:10]):
        print(f"\nProblem {problem['problem_id']}:")
        print(f"  정답: {problem['ground_truth']}")
        print(f"  첫 정답 percentage: {problem['first_correct_percentage']}%")
        print(f"  지속 정답 시작: {problem['always_correct_from']}%")

        # 각 percentage별 결과
        correct_pattern = []
        for percentage in range(10, 101, 10):
            if percentage in problem["percentages"]:
                is_correct = problem["percentages"][percentage]["correct"]
                correct_pattern.append("O" if is_correct else "X")
            else:
                correct_pattern.append("-")

        print(f"  Pattern (10%-100%): {' '.join(correct_pattern)}")


def save_problem_results(problem_results: List[Dict], output_path: str):
    """문제별 결과를 CSV 파일로 저장"""
    problem_data = []

    for problem in problem_results:
        row = {
            "problem_id": problem["problem_id"],
            "ground_truth": problem["ground_truth"],
            "first_correct_percentage": problem["first_correct_percentage"],
            "always_correct_from": problem["always_correct_from"],
        }

        # 각 percentage별 결과 추가
        for percentage in range(10, 101, 10):
            if percentage in problem["percentages"]:
                row[f"correct_{percentage}"] = problem["percentages"][percentage][
                    "correct"
                ]
                row[f"prediction_{percentage}"] = problem["percentages"][percentage][
                    "prediction"
                ]
                row[f"generated_length_{percentage}"] = problem["percentages"][
                    percentage
                ]["generated_length"]
                row[f"total_length_{percentage}"] = problem["percentages"][percentage][
                    "total_length"
                ]
                row[f"think_length_{percentage}"] = problem["percentages"][percentage][
                    "think_length"
                ]
            else:
                row[f"correct_{percentage}"] = None
                row[f"prediction_{percentage}"] = None
                row[f"generated_length_{percentage}"] = None
                row[f"total_length_{percentage}"] = None
                row[f"think_length_{percentage}"] = None

        problem_data.append(row)

    df = pd.DataFrame(problem_data)
    df.to_csv(output_path, index=False)
    print(f"\n문제별 결과가 '{output_path}'에 저장되었습니다.")


def plot_first_correct_distribution(problem_results: List[Dict], output_path: str):
    """첫 번째 정답 percentage 분포 그래프"""
    first_correct_percentages = []
    for problem in problem_results:
        if problem["first_correct_percentage"] is not None:
            first_correct_percentages.append(problem["first_correct_percentage"])

    if not first_correct_percentages:
        print("정답을 맞춘 문제가 없어 분포 그래프를 생성할 수 없습니다.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(
        first_correct_percentages, bins=range(0, 101, 10), alpha=0.7, edgecolor="black"
    )
    plt.xlabel("First Correct Percentage (%)")
    plt.ylabel("Number of Problems")
    plt.title("Distribution of First Correct Percentage")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(10, 101, 10))

    # 통계 정보 추가
    mean_first = sum(first_correct_percentages) / len(first_correct_percentages)
    plt.axvline(
        mean_first, color="red", linestyle="--", label=f"Mean: {mean_first:.1f}%"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"첫 정답 분포 그래프가 '{output_path}'에 저장되었습니다.")


def analyze_problem_difficulty(problem_results: List[Dict]) -> Dict:
    """문제 난이도 분석"""
    difficulty_analysis = {
        "easy": [],  # 10-30%에서 맞춤
        "medium": [],  # 40-70%에서 맞춘
        "hard": [],  # 80-100%에서 맞춤
        "unsolved": [],  # 끝까지 못 맞춤
    }

    for problem in problem_results:
        first_correct = problem["first_correct_percentage"]
        if first_correct is None:
            difficulty_analysis["unsolved"].append(problem["problem_id"])
        elif first_correct <= 30:
            difficulty_analysis["easy"].append(problem["problem_id"])
        elif first_correct <= 70:
            difficulty_analysis["medium"].append(problem["problem_id"])
        else:
            difficulty_analysis["hard"].append(problem["problem_id"])

    return difficulty_analysis


def save_problem_detailed_info(problem_results: List[Dict], output_path: str):
    """문제별 상세 정보를 텍스트 파일로 저장 (100%에서 틀린 문제들만)"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== 문제별 상세 정보 (100%에서 틀린 문제들만) ===\n\n")
        f.write(f"총 {len(problem_results)}개 문제 (100%에서 맞은 문제들은 제외됨)\n\n")

        for problem in problem_results:
            f.write(f"Problem {problem['problem_id']}:\n")
            f.write(f"  정답: {problem['ground_truth']}\n")
            f.write(f"  첫 정답 percentage: {problem['first_correct_percentage']}%\n")
            f.write(f"  지속 정답 시작: {problem['always_correct_from']}%\n")

            # 각 percentage별 결과
            correct_pattern = []
            for percentage in range(10, 101, 10):
                if percentage in problem["percentages"]:
                    is_correct = problem["percentages"][percentage]["correct"]
                    correct_pattern.append("O" if is_correct else "X")
                else:
                    correct_pattern.append("-")

            f.write(f"  Pattern (10%-100%): {' '.join(correct_pattern)}\n")

            # 각 percentage별 상세 결과
            f.write("  상세 결과:\n")
            for percentage in range(10, 101, 10):
                if percentage in problem["percentages"]:
                    result = problem["percentages"][percentage]
                    f.write(
                        f"    {percentage}%: {'정답' if result['correct'] else '오답'} "
                        f"(예측: {result['prediction']}, GT: {problem['ground_truth']}, "
                        f"생성길이: {result['generated_length']} tokens, "
                        f"전체길이: {result['total_length']} tokens)\n"
                    )

            # 생성 길이 요약
            generated_lengths = [
                problem["percentages"][p]["generated_length"]
                for p in range(10, 101, 10)
                if p in problem["percentages"]
            ]
            total_lengths = [
                problem["percentages"][p]["total_length"]
                for p in range(10, 101, 10)
                if p in problem["percentages"]
            ]

            if generated_lengths and total_lengths:
                avg_gen_length = sum(generated_lengths) / len(generated_lengths)
                min_gen_length = min(generated_lengths)
                max_gen_length = max(generated_lengths)

                avg_total_length = sum(total_lengths) / len(total_lengths)
                min_total_length = min(total_lengths)
                max_total_length = max(total_lengths)

                f.write(f"  생성 길이 요약:\n")
                f.write(
                    f"    순수 생성 - 평균: {avg_gen_length:.1f}, 최소: {min_gen_length}, 최대: {max_gen_length} tokens\n"
                )
                f.write(
                    f"    전체 응답 - 평균: {avg_total_length:.1f}, 최소: {min_total_length}, 최대: {max_total_length} tokens\n"
                )

            f.write("\n")


def save_problem_summary_stats(problem_results: List[Dict], output_path: str):
    """문제별 통계 요약을 JSON 파일로 저장"""
    # 첫 번째로 맞춘 percentage 분포
    first_correct_dist = {}
    always_correct_dist = {}
    never_correct_count = 0

    for problem in problem_results:
        if problem["first_correct_percentage"] is not None:
            first_percentage = problem["first_correct_percentage"]
            first_correct_dist[first_percentage] = (
                first_correct_dist.get(first_percentage, 0) + 1
            )
        else:
            never_correct_count += 1

        if problem["always_correct_from"] is not None:
            always_percentage = problem["always_correct_from"]
            always_correct_dist[always_percentage] = (
                always_correct_dist.get(always_percentage, 0) + 1
            )

    # 통계 계산
    first_correct_percentages = [
        p["first_correct_percentage"]
        for p in problem_results
        if p["first_correct_percentage"] is not None
    ]

    stats = {
        "total_problems": len(problem_results),
        "solved_problems": len(first_correct_percentages),
        "never_solved": never_correct_count,
        "first_correct_distribution": first_correct_dist,
        "always_correct_distribution": always_correct_dist,
        "statistics": {
            "mean_first_correct": (
                sum(first_correct_percentages) / len(first_correct_percentages)
                if first_correct_percentages
                else None
            ),
            "median_first_correct": (
                sorted(first_correct_percentages)[len(first_correct_percentages) // 2]
                if first_correct_percentages
                else None
            ),
            "solve_rate": (
                len(first_correct_percentages) / len(problem_results)
                if problem_results
                else 0
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"문제별 통계 요약이 '{output_path}'에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate early exit results")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to overthinking_results.jsonl file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name for tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # 데이터 로드
    print(f"Loading data from {args.input_path}...")
    data = read_jsonl(args.input_path)

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 성능 평가
    performance_summary, percentage_results, problem_results = (
        evaluate_early_exit_results(data, tokenizer)
    )

    # 결과 출력
    print_performance_summary(performance_summary)
    print_problem_analysis(problem_results)

    # 결과 저장
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    # 상세 결과 저장
    detailed_csv_path = os.path.join(args.output_dir, "early_exit_detailed_results.csv")
    save_detailed_results(percentage_results, detailed_csv_path)

    # 문제별 결과 저장
    problem_csv_path = os.path.join(args.output_dir, "problem_wise_results.csv")
    save_problem_results(problem_results, problem_csv_path)

    # 문제별 상세 정보 파일로 저장
    problem_detailed_path = os.path.join(args.output_dir, "problem_detailed_info.txt")
    save_problem_detailed_info(problem_results, problem_detailed_path)

    # 문제별 통계 요약 저장
    problem_stats_path = os.path.join(args.output_dir, "problem_statistics.json")
    save_problem_summary_stats(problem_results, problem_stats_path)

    # 첫 정답 분포 그래프
    first_correct_plot_path = os.path.join(
        args.output_dir, "first_correct_distribution.png"
    )
    plot_first_correct_distribution(problem_results, first_correct_plot_path)

    # 문제 난이도 분석
    difficulty_analysis = analyze_problem_difficulty(problem_results)
    difficulty_path = os.path.join(args.output_dir, "difficulty_analysis.json")
    with open(difficulty_path, "w") as f:
        json.dump(difficulty_analysis, f, indent=2)
    print(f"난이도 분석이 '{difficulty_path}'에 저장되었습니다.")

    # 성능 요약 저장
    summary_path = os.path.join(args.output_dir, "early_exit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(performance_summary, f, indent=2)
    print(f"성능 요약이 '{summary_path}'에 저장되었습니다.")

    # 그래프 생성
    plot_path = os.path.join(args.output_dir, "early_exit_performance_curve.png")
    plot_performance_curve(performance_summary, plot_path)

    # 문제별 분석 결과 출력
    print_problem_analysis(problem_results)

    # 난이도 분석 결과
    difficulty_analysis = analyze_problem_difficulty(problem_results)
    print("\n=== 문제 난이도 분석 결과 ===")
    for level, problems in difficulty_analysis.items():
        print(f"{level.capitalize()} 문제: {len(problems)}개")

    # 문제별 상세 정보 저장
    detailed_info_path = os.path.join(args.output_dir, "problem_detailed_info.txt")
    save_problem_detailed_info(problem_results, detailed_info_path)

    # 문제별 통계 요약 저장
    summary_stats_path = os.path.join(args.output_dir, "problem_summary_stats.json")
    save_problem_summary_stats(problem_results, summary_stats_path)


if __name__ == "__main__":
    main()
