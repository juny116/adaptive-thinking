import json
import numpy as np
from transformers import AutoTokenizer

# 1) paths
VANILLA_PATH = "results/math_500/qwen_base_0.jsonl"
OURS_PATH    = "results/math_500/qwen_wait_no_sys_1.jsonl"

# 2) load tokenizer (Qwen3-32B)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-32B",
    trust_remote_code=True
)

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def assistant_text(example):
    # messages: [user, assistant]
    # assistant content is usually messages[1]["content"]
    return example["messages"][1]["content"]

def token_len(text):
    return len(tokenizer.encode(text))

# 3) load data
vanilla = read_jsonl(VANILLA_PATH)
ours    = read_jsonl(OURS_PATH)

# sanity check matching indices
vanilla_map = {ex["index"]: ex for ex in vanilla}
ours_map    = {ex["index"]: ex for ex in ours}
common_idx  = sorted(set(vanilla_map.keys()) & set(ours_map.keys()))

print("num common problems:", len(common_idx))

# 4) compute per-problem lengths
len_v = []
len_o = []
for idx in common_idx:
    tv = assistant_text(vanilla_map[idx])
    to = assistant_text(ours_map[idx])

    len_v.append(token_len(tv))
    len_o.append(token_len(to))

len_v = np.array(len_v)
len_o = np.array(len_o)

# 5) differences (vanilla - ours)
D = len_v - len_o
print("mean vanilla length:", len_v.mean())
print("mean ours length   :", len_o.mean())
print("mean reduction     :", D.mean())

# 6) paired bootstrap CI
def bootstrap_mean_ci(D, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(D)
    means = np.empty(B)
    for b in range(B):
        sample = rng.choice(D, size=N, replace=True)
        means[b] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return low, high, means

ci_low, ci_high, means = bootstrap_mean_ci(D, B=10000)
print(f"95% bootstrap CI for mean reduction: [{ci_low:.2f}, {ci_high:.2f}]")

# optional: sign test style summary
pos = (D > 0).sum()
neg = (D < 0).sum()
zero = (D == 0).sum()
print("per-problem reduction sign:",
      f"+:{pos}  -:{neg}  0:{zero}")
