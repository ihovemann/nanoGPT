import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_summary(name, filepath):
    with open(filepath) as f:
        lines = f.readlines()

    config = {}
    for line in lines:
        for key in ["learning_rate", "n_layer", "n_embd", "block_size", "dropout", "max_iters"]:
            if f"{key} = " in line:
                config[key] = line.split("=")[1].strip()

    train_loss = val_loss = None
    for line in lines:
        m = re.search(r"step \d+: train loss ([\d.]+), val loss ([\d.]+)", line)
        if m:
            train_loss = float(m.group(1))
            val_loss = float(m.group(2))

    total_ms = 0
    for line in lines:
        m = re.search(r"iter \d+: loss [\d.]+, time ([\d.]+)ms", line)
        if m:
            total_ms += float(m.group(1))

    return {
        "Experiment": name,
        "LR":         config["learning_rate"],
        "Layers":     config["n_layer"],
        "Embd":       config["n_embd"],
        "Block":      config["block_size"],
        "Dropout":    config["dropout"],
        "Iters":      config["max_iters"],
        "Train Loss": train_loss,
        "Val Loss":   val_loss,
        "Time (min)": round(total_ms / 60000, 1),
    }


def parse_losses(filepath):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    with open(filepath) as f:
        for line in f:
            m = re.search(r"iter (\d+): loss ([\d.]+)", line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))

            m = re.search(r"step (\d+): train loss [\d.]+, val loss ([\d.]+)", line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))

    return train_steps, train_losses, val_steps, val_losses


runs = {
    "baseline":        "logs/baseline.log",
    "model_width_128": "logs/model_width_128.log",
}

# Summary table
rows = [parse_summary(name, path) for name, path in runs.items()]
df = pd.DataFrame(rows)
print(df)

with open("summary.md", "w") as f:
    f.write(df.to_markdown(index=False))

# Plots
fig_train, ax_train = plt.subplots()
fig_val, ax_val = plt.subplots()

for name, path in runs.items():
    train_steps, train_losses, val_steps, val_losses = parse_losses(path)
    ax_train.plot(train_steps, train_losses, label=name)
    ax_val.plot(val_steps, val_losses, label=name)

ax_train.set_title("Training Loss vs Step")
ax_train.set_xlabel("Step")
ax_train.set_ylabel("Loss")
ax_train.legend()

ax_val.set_title("Validation Loss vs Step")
ax_val.set_xlabel("Step")
ax_val.set_ylabel("Loss")
ax_val.legend()

fig_train.savefig("train_loss.png", dpi=150, bbox_inches="tight")
fig_val.savefig("val_loss.png", dpi=150, bbox_inches="tight")

plt.show()