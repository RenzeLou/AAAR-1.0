import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.lines as mlines

def darken_color(hex_color, factor=0.7):
    """
    Darken the given hex color by the specified factor.
    
    :param hex_color: str, Hex color code (e.g., '#5CAB7D').
    :param factor: float, Factor by which to darken the color (e.g., 0.7 means 70% of the original color).
    :return: str, New hex color code.
    """
    # Ensure the factor is between 0 and 1
    factor = max(0, min(1, factor))
    
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Darken the color
    darkened_rgb = tuple(int(c * factor) for c in rgb)
    
    # Convert back to hex
    return "#{:02X}{:02X}{:02X}".format(*darkened_rgb)


# Enable the usage of Times New Roman font
plt.rcParams["font.family"] = "Times New Roman"


# ============================== Experiment score ==============================


# Define scales and performances
x_scales = ["0.1k", "0.5k", "1k", "3k", "5k", "8k", "10k"]
# Adjust space by slightly reducing the last number
# x_scales_numbers = np.array([100, 500, 1000, 3000, 5000, 8000, 10000, 12000, 15000])
# make the x_scale_numbers equal spaced
x_scales_numbers = np.array([100, 200, 300, 400, 500, 600, 700])
gpt4_turbo = [44.71,	50.46,	51.91,	53.08,	54.57,	56.05,	56.22]
gpt4o = [45.63,	46.37,	49.87,	53.00,	54.31,	55.87,	55.76]
qwen = [36.21,	41.24,	43.14,	42.58,	44.79,	42.82,	42.28]
llama = [30.73,	36.17,	39.22,	39.37,	41.98,	42.78,	42.37]


datasets = {
    'GPT-4o': (gpt4o, 'x', "#C65146", '-'), # '#87484A' "#C89EC4"
    'GPT-4-Turbo': (gpt4_turbo, 'o', "#30A9DE", "-"),  # "#759AB3"  "#67D5B5"
    'Qwen-2.5': (qwen, '^', "#5CAB7D", '-.'),  # "#83AF98"  "#E53A40"
    'Llama-3.1': (llama, 'D', "#9055A2", '-.'),  # "#D7B06F"  "#AACD6E"
}

# Create legend handles
legend_handles = []

# Create a figure and axis with larger text
plt.figure(figsize=(10, 6.8))
for name, (performance, marker, color, line_style) in datasets.items():
    # Remove missing data points
    x, y = zip(*[(i, perf) for i, perf in zip(x_scales_numbers, performance) if not np.isnan(perf)])
    
    # Make the line smoother using UnivariateSpline
    spl = UnivariateSpline(x, y, s=1, k=2)  # You can adjust k (degree of the spline) and s (smoothing factor)
    xs = np.linspace(min(x), max(x), 1000)
    ys = spl(xs)
    
    # plt.plot(xs, ys, label=name, linewidth=2.7)
    line = plt.plot(xs, ys, label=name, linewidth=2.6, color=color, linestyle=line_style)[0]
    plt.scatter(x, y, marker=marker, color=color, s=50, edgecolor=darken_color(color), zorder=5)
        
    # Annotate points
    # for xp, yp in zip(x, y):
    #     plt.annotate(f'{yp:.2f}', (xp, yp), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)
    
    # Add custom legend entry with both line and marker
    legend_handles.append(mlines.Line2D([], [], color=line.get_color(), marker=marker, linestyle=line_style, label=name, linewidth=2.6))

# Setting the x-axis as described in task
plt.xticks(x_scales_numbers, x_scales, fontsize=14)
# setting the font size of y-axis
plt.yticks(fontsize=14)

# Create labels and title with larger font size
plt.xlabel("Input Context Length (# of Words)", fontsize=22)
plt.ylabel("Experiment Design (F1)", fontsize=20) # across 4 benchmarks
# plt.title("SubTask 1 --- Input Context Scaling Trend", fontsize=26)


# Add grid to the figure, with a light gray color and a dashed line style
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)  # higher alpha value for darker grid lines

# Show the legend with Times New Roman font (also show the marker)
# plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper left', markerscale=1.5)
# Use custom legend handles
plt.legend(handles=legend_handles, prop={'family': 'Times New Roman', 'size': 15}, loc='upper left')

# Making the plot layout tight
plt.tight_layout()

# Save the plot as a PDF
plt.savefig("subtask2_input_context_scaling.f1.pdf", format='pdf', bbox_inches='tight')





# ============================== Explanation score ==============================

# Define scales and performances
x_scales = ["0.1k", "0.5k", "1k", "3k", "5k", "8k", "10k"]
# Adjust space by slightly reducing the last number
# x_scales_numbers = np.array([100, 500, 1000, 3000, 5000, 8000, 10000, 12000, 15000])
# make the x_scale_numbers equal spaced
x_scales_numbers = np.array([100, 200, 300, 400, 500, 600, 700])
gpt4_turbo = [54.73,	53.47,	55.12,	55.87,	55.81,	56.67,	56.48]
gpt4o = [58.79,	58.08,	57.37,	58.24,	57.97,	57.48,	58.12]
qwen = [51.17,	50.78,	50.41,	51.31,	50.76,	50.22,	49.56]
llama = [49.11,	48.04,	48.70,	49.58,	48.59,	49.31,	48.29]


datasets = {
    'GPT-4o': (gpt4o, 'x', "#C65146", '-'), # '#87484A' "#C89EC4"
    'GPT-4-Turbo': (gpt4_turbo, 'o', "#30A9DE", "-"),  # "#759AB3"  "#67D5B5"
    'Qwen-2.5': (qwen, '^', "#5CAB7D", '-.'),  # "#83AF98"  "#E53A40"
    'Llama-3.1': (llama, 'D', "#9055A2", '-.'),  # "#D7B06F"  "#AACD6E"
}

# Create legend handles
legend_handles = []

# Create a figure and axis with larger text
plt.figure(figsize=(10, 6.8))
for name, (performance, marker, color, line_style) in datasets.items():
    # Remove missing data points
    x, y = zip(*[(i, perf) for i, perf in zip(x_scales_numbers, performance) if not np.isnan(perf)])
    
    # Make the line smoother using UnivariateSpline
    spl = UnivariateSpline(x, y, s=1.0, k=2)  # You can adjust k (degree of the spline) and s (smoothing factor)
    xs = np.linspace(min(x), max(x), 1000)
    ys = spl(xs)
    
    # plt.plot(xs, ys, label=name, linewidth=2.7)
    line = plt.plot(xs, ys, label=name, linewidth=2.6, color=color, linestyle=line_style)[0]
    plt.scatter(x, y, marker=marker, color=color, s=50, edgecolor=darken_color(color), zorder=5)
        
    # Annotate points
    # for xp, yp in zip(x, y):
    #     plt.annotate(f'{yp:.2f}', (xp, yp), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)
    
    # Add custom legend entry with both line and marker
    legend_handles.append(mlines.Line2D([], [], color=line.get_color(), marker=marker, linestyle=line_style, label=name, linewidth=2.6))

# Setting the x-axis as described in task
plt.xticks(x_scales_numbers, x_scales, fontsize=14)
# setting the font size of y-axis
plt.yticks(fontsize=14)

# Create labels and title with larger font size
plt.xlabel("Input Context Length (# of Words)", fontsize=22)
plt.ylabel("Explanation Gen. (Soft-Match)", fontsize=20) # across 4 benchmarks
# plt.title("SubTask 1 --- Input Context Scaling Trend", fontsize=26)


# Add grid to the figure, with a light gray color and a dashed line style
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)  # higher alpha value for darker grid lines

# Show the legend with Times New Roman font (also show the marker)
# plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper left', markerscale=1.5)
# Use custom legend handles
plt.legend(handles=legend_handles, prop={'family': 'Times New Roman', 'size': 15}, loc='upper left')

# Making the plot layout tight
plt.tight_layout()

# Save the plot as a PDF
plt.savefig("subtask2_input_context_scaling.soft_match.pdf", format='pdf', bbox_inches='tight')
