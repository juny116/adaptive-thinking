"""Plot self-reported and internal confidence correlations."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from adaptive_thinking.data.jsonl import read_jsonl

    data_path = args.input
    test_data = read_jsonl(data_path)
    # tokenizer=  AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    num_wait = 0
    num_alt = 0
    confidence_class = {
        "Almost no chance": 0.5,
        "Highly unlikely": 1.5,
        "Chances are slight": 2.5,
        "Unlikely": 3.5,
        "Less than even": 4.5,
        "Better than even": 5.5,
        "Likely": 6.5,
        "Very good chance": 7.5,
        "Highly likely": 8.5,
        "Almost certain": 9.5,
    }

    self_conf = []
    met_conf = []

    for i, d in enumerate(test_data):
        content = d["messages"][-1]["content"]

        confidences_self = d["confidence_self"]
        # convert confidence_self to a conf value
        confidences_self_converted = [
            confidence_class.get(conf, 0) for conf in confidences_self
        ]
        confidences_met = d["confidence_met"]

        if len(confidences_self_converted) == len(confidences_met):
            for s, m in zip(confidences_self_converted, confidences_met):
                self_conf.append(s)
                met_conf.append(m)

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import pearsonr

    if len(self_conf) < 2:
        raise ValueError("Need at least two paired confidence values")
    # Convert to numpy arrays
    self_conf = np.array(self_conf)
    met_conf = np.array(met_conf)
    # Normalize values to [0,1] for better visual comparison
    self_conf_norm = (self_conf - self_conf.min()) / (
        self_conf.max() - self_conf.min() + 1e-8
    )
    met_conf_norm = (met_conf - met_conf.min()) / (
        met_conf.max() - met_conf.min() + 1e-8
    )

    # Plotting
    plt.figure(figsize=(8, 6))
    # Scatter of normalized confidence values
    plt.scatter(self_conf_norm, met_conf_norm, alpha=0.6)
    # Overlay regression line and 95% confidence interval using seaborn
    sns.regplot(
        x=self_conf_norm,
        y=met_conf_norm,
        scatter=False,
        ci=95,
        color="red",
        line_kws={"label": "95% CI fit"},
    )
    # Update axis labels to reflect normalization
    plt.xlabel("Normalized Self-assessed Confidence")
    plt.ylabel("Normalized Internal Confidence")
    plt.title("Normalized Self-assessed vs Internal Confidence")
    # Annotate Pearson correlation and p-value
    corr_val, p_val = pearsonr(self_conf_norm, met_conf_norm)
    plt.text(
        0.05,
        0.95,
        f"Pearson r={corr_val:.3f}\nP-value={p_val:.3g}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.grid(True)
    plt.savefig(args.output)
    plt.close()
    print(f"Plot saved as {args.output}")

    # Correlation
    corr = np.corrcoef(self_conf, met_conf)[0, 1]
    print(f"Correlation: {corr:.3f}")

    # Calculate Pearson correlation and p-value
    corr, p_value = pearsonr(self_conf, met_conf)
    print(f"Pearson correlation: {corr:.3f}")
    print(f"P-value: {p_value:.4g}")


if __name__ == "__main__":
    main()
