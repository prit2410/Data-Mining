import matplotlib.pyplot as plt
import numpy as np

# Data: Risk, Probability, Impact
risks = [
    ("Integration challenges with third-party APIs", 40, 30000),
    ("AI model performance issues", 65, 50000),
    ("Inaccurate real-time travel updates", 40, 15000),
    ("Team conflicts and lack of collaboration", 70, 25000),
    ("Dependency on cloud services", 20, 20000),
    ("High costs from paid APIs", 25, 10000),
]

# Convert percentages to a 0-1 scale
probability = [p / 100 for _, p, _ in risks]
impact = [i for _, _, i in risks]

# Define risk categories based on probability
def get_risk_color(prob):
    if prob <= 0.35:
        return "white"  # Low risk (Recognize)
    elif prob <= 0.60:
        return "green"  # Moderate risk (Mitigate)
    else:
        return "black"  # High risk (Eliminate)

colors = [get_risk_color(p) for p in probability]

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 5))

# Background color regions
ax.axhspan(0, 0.35, facecolor="white", alpha=0.5, label="Recognize")
ax.axhspan(0.35, 0.60, facecolor="green", alpha=0.3, label="Mitigate")
ax.axhspan(0.60, 1.0, facecolor="black", alpha=0.3, label="Eliminate")

# Scatter plot
for (label, p, i), color in zip(risks, colors):
    ax.scatter(i, p / 100, color=color, edgecolors="black", s=100)
    ax.text(i, p / 100, label, fontsize=10, ha="right", va="bottom", color="white" if color == "black" else "black")

# Labels and title
ax.set_xlabel("IMPACT ($)", fontsize=12, fontweight="bold")
ax.set_ylabel("PROBABILITY", fontsize=12, fontweight="bold")
ax.set_title("Risk Assessment Matrix", fontsize=14, fontweight="bold")

# Formatting the axes
ax.set_xticks(np.linspace(0, 60000, 5))
ax.set_yticks([0, 0.35, 0.60, 1.0])
ax.set_yticklabels(["Low", "Medium", "High", "Very High"])
ax.set_xlim(0, 60000)
ax.set_ylim(0, 1)

# Show grid
ax.grid(True, linestyle="--", alpha=0.6)

plt.show()
