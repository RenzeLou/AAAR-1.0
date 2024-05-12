import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.lines as mlines

# Enable the usage of Times New Roman font
plt.rcParams["font.family"] = "Times New Roman"

# Define scales and performances
x_scales = ["100", "300", "500", "700", "900", "1100", "1300", "1500"]
# Adjust space by slightly reducing the last number
x_scales_numbers = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500])
gpt4_turbo = [24.88, 34.89, 39.37, 39.08, 40.89, 41.27, 41.56, 41.37]
# claude_3 = [31.44, 32.62, 32.84, 32.71, 34.02 , 33.2]

datasets = {
    'GPT-4-Turbo': (gpt4_turbo, 'o', "#67D5B5"),  # square marker
    # 'Self-Inst.': (self_inst, 'D', "#C89EC4"),  # diamond marker
    # 'Unnatural Inst.': (unnatural_inst, '^', "#E53A40"),  # triangle_up marker
    # 'Dynosaur': (dynosaur, 'x', "#30A9DE"),  # x marker
    # 'Alpaca': (alpaca, 's', "#791E94"),  # star marker
}

# Create legend handles
legend_handles = []

# Create a figure and axis with larger text
plt.figure(figsize=(10, 10))
for name, (performance, marker, color) in datasets.items():
    # Remove missing data points
    x, y = zip(*[(i, perf) for i, perf in zip(x_scales_numbers, performance) if not np.isnan(perf)])
    # Make the line smoother using UnivariateSpline
    spl = UnivariateSpline(x, y, s=0.2, k=2)  # You can adjust k (degree of the spline) and s (smoothing factor)
    xs = np.linspace(min(x), max(x), 1000)
    ys = spl(xs)
    # plt.plot(xs, ys, label=name, linewidth=2.7)
    line = plt.plot(xs, ys, label=name, linewidth=2.6, color=color)[0]
    plt.scatter(x, y, marker=marker, color=color, s=50)
    # Annotate points
    for xp, yp in zip(x, y):
        plt.annotate(f'{yp:.2f}', (xp, yp), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)
    # Add custom legend entry with both line and marker
    legend_handles.append(mlines.Line2D([], [], color=line.get_color(), marker=marker, linestyle='-', label=name))

# Setting the x-axis as described in task
plt.xticks(x_scales_numbers, x_scales, fontsize=14)
# setting the font size of y-axis
plt.yticks(fontsize=14)

# Create labels and title with larger font size
plt.xlabel("Input Context Length (In Words)", fontsize=22)
plt.ylabel("Classification Accuracy", fontsize=20) # across 4 benchmarks
plt.title("SubTask 1 --- Input Context Scaling Trend", fontsize=26)


# Add grid to the figure, with a light gray color and a dashed line style
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the legend with Times New Roman font (also show the marker)
# plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper left', markerscale=1.5)
# Use custom legend handles
plt.legend(handles=legend_handles, prop={'family': 'Times New Roman', 'size': 15}, loc='upper left')

# Making the plot layout tight
plt.tight_layout()

# Save the plot as a PDF
plt.savefig("subtask1_input_context_scaling.pdf", format='pdf', bbox_inches='tight')