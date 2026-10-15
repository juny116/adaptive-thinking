"""Command dispatcher for generation, evaluation and data tools."""

import argparse
import runpy
import sys

COMMANDS = {
    "run": {
        "confidence_continue": "strategies.confidence_continue",
        "deer": "strategies.deer",
        "baseline": "strategies.baseline",
        "adaptive": "strategies.adaptive",
        "thinking": "strategies.thinking",
        "direct": "strategies.direct",
        "confidence_prompt_ablation": "strategies.variants.confidence_prompt_ablation",
        "segment_confidence": "strategies.variants.segment_confidence",
        "fixed_length": "strategies.variants.fixed_length",
        "alternative_stop": "strategies.variants.alternative_stop",
        "fixed_length_completion": "strategies.variants.fixed_length_completion",
        "hesitation_stops": "strategies.variants.hesitation_stops",
        "segment_confidence_control": "strategies.variants.segment_confidence_control",
        "unlimited_continuation": "strategies.variants.unlimited_continuation",
        "wait_alternative_stops": "strategies.variants.wait_alternative_stops",
        "deer_token_budget": "strategies.variants.deer_token_budget",
        "wait_stop": "strategies.variants.wait_stop",
        "wait_only": "strategies.variants.wait_only",
    },
    "analyze": {
        "length_comparison": "analysis.length_comparison",
        "confidence_correlation": "analysis.confidence_correlation",
        "confidence_statistics": "analysis.confidence_statistics",
        "compare_runs": "analysis.compare_runs",
        "compare_files": "analysis.compare_files",
        "compare_settings": "analysis.compare_settings",
        "plot_benchmarks": "analysis.plot_benchmarks",
    },
    "data": {
        "split_reasoning": "data.split_reasoning",
        "math500": "data.math500",
        "math_knowledge": "data.math_knowledge",
        "amc2023": "data.amc2023",
        "strategyqa": "data.strategyqa",
        "normalize_answers": "data.normalize_answers",
        "aime2025": "data.aime2025",
        "aime2024": "data.aime2024",
        "math": "data.math",
        "olympiadbench": "data.olympiadbench",
        "prepare_gpqa": "data.prepare_gpqa",
    },
    "evaluate": {"answers": "evaluation.score_file"},
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Adaptive reasoning generation and analysis"
    )
    parser.add_argument("command", choices=[*COMMANDS, "list"])
    parser.add_argument("tool", nargs="?")
    parser.add_argument("tool_args", nargs=argparse.REMAINDER)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        print("\nUse 'python -m adaptive_thinking list' to list tools.")
        return
    if args[0] == "list":
        for group, tools in COMMANDS.items():
            print(group + ":")
            for name in sorted(tools):
                print("  " + name)
        return
    if args[0] not in COMMANDS:
        parser.error(f"unknown command: {args[0]}")
    group = args[0]
    if len(args) < 2 or args[1] in ("-h", "--help"):
        print(group + ": " + ", ".join(sorted(COMMANDS[group])))
        return
    name = args[1].replace("-", "_")
    if name not in COMMANDS[group]:
        parser.error(f"unknown {group} tool: {args[1]}")
    module = "adaptive_thinking." + COMMANDS[group][name]
    old_argv = sys.argv
    try:
        sys.argv = [module, *args[2:]]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv
