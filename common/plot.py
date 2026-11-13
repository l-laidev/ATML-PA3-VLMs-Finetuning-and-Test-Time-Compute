import torch
import matplotlib.pyplot as plt
import pandas as pd


model = "lora_r16"
base = f"checkpoints/task1/{model}/checkpoint_update_1250_examples_40000/training_state.pt"
hist = torch.load(base, map_location="cpu", weights_only=False)["logger_state"]["history"]


df = pd.DataFrame(hist)
metrics = hist[0].keys()
metrics = list(filter(lambda m: not any([kw in m for kw in ['step', 'is_skipped']]) and len(df[m].dropna()) != 0, metrics))

f, ax = plt.subplots(1, len(list(metrics)), figsize=(15,4))
ax = ax.ravel()
window_size = 30

print(metrics)
for m, a in zip(metrics, ax):
    print("Plotting: ", m)

    df.plot(x="step", y=m, ax=a, alpha=0.3, color="gray", legend=False)
    
    smoothed_col = f"{m}_smoothed"
    df[smoothed_col] = df[m].rolling(window=window_size, min_periods=1).mean()
    df.plot(x="step", y=smoothed_col, ax=a, color="blue", linewidth=2, legend=False)
    
    a.set_title(m)

plt.tight_layout()
plt.savefig("test.png")
