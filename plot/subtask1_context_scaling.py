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

# Define scales and performances
x_scales = ["100", "300", "500", "700", "900", "1100", "1300", "1500"]
# Adjust space by slightly reducing the last number
x_scales_numbers = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500])
gpt4_turbo = [24.88, 34.89, 39.37, 39.08, 40.89, 41.27, 41.56, 41.37]
gpt4o = [30.79, 39.46, 39.94, 41.84, 42.99, 44.23, 44.42, 43.46]
qwen = [31.07,37.36,	35.93,	33.36,	30.12,	29.16,		27.74,	26.26]
llama = [29.64,	39.08,	38.13,	38.6,	39.084,	38.22,		37.51,	38.13]
# claude_3 = [31.44, 32.62, 32.84, 32.71, 34.02 , 33.2]


datasets = {
    'GPT-4o': (gpt4o, 'x', "#C65146", '-'), # '#87484A' "#C89EC4"
    'GPT-4-Turbo': (gpt4_turbo, 'o', "#30A9DE", "-"),  # "#759AB3"  "#67D5B5"
    'Qwen-2.5': (qwen, '^', "#5CAB7D", '-.'),  # "#83AF98"  "#E53A40"
    'Llama-3.1': (llama, 'D', "#9055A2", '-.'),  # "#D7B06F"  "#AACD6E"
    # 'Alpaca': (alpaca, 's', "#791E94"),  # star marker
}

# Create legend handles
legend_handles = []

# Create a figure and axis with larger text
plt.figure(figsize=(10, 7.3))
for name, (performance, marker, color, line_style) in datasets.items():
    # Remove missing data points
    x, y = zip(*[(i, perf) for i, perf in zip(x_scales_numbers, performance) if not np.isnan(perf)])
    
    # Make the line smoother using UnivariateSpline
    spl = UnivariateSpline(x, y, s=0.9, k=2)  # You can adjust k (degree of the spline) and s (smoothing factor)
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
plt.ylabel("Accuracy", fontsize=20) # across 4 benchmarks
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
plt.savefig("subtask1_input_context_scaling.pdf", format='pdf', bbox_inches='tight')













# datasets = {
#     'GPT-4o': (gpt4o, 'x', "#C65146", '-'), # '#87484A' "#C89EC4"
#     'GPT-4-Turbo': (gpt4_turbo, 'o', "#30A9DE", "-"),  # "#759AB3"  "#67D5B5"
#     'Qwen-2.5': (qwen, '^', "#a5d296", '-.'),  # "#83AF98"  "#E53A40"
#     'Llama-3.1': (llama, 'D', "#9055A2", '-.'),  # "#D7B06F"  "#AACD6E"
#     # 'Alpaca': (alpaca, 's', "#791E94"),  # star marker
# }

# # Create legend handles
# legend_handles = []

# # Create a figure and axis with larger text
# plt.figure(figsize=(10, 8))
# for name, (performance, marker, color, line_style) in datasets.items():
#     # Remove missing data points
#     x, y = zip(*[(i, perf) for i, perf in zip(x_scales_numbers, performance) if not np.isnan(perf)])
    
#     # Make the line smoother using UnivariateSpline
#     spl = UnivariateSpline(x, y, s=0.9, k=2)  # You can adjust k (degree of the spline) and s (smoothing factor)
#     xs = np.linspace(min(x), max(x), 1000)
#     ys = spl(xs)
    
#     # Define the error margin (e.g., ±5% of y max or your specific range of interest)
#     error_margin = 0.03 * max(y)
#     # Use the actual data `y` for range with applied margin
#     spl_upper = UnivariateSpline(x, np.array(y) + error_margin, s=0.9, k=2)
#     spl_lower = UnivariateSpline(x, np.array(y) - error_margin, s=0.9, k=2)
#     # ys_upper = y + error_margin
#     # ys_lower = y - error_margin
#     ys_upper = spl_upper(xs)
#     ys_lower = spl_lower(xs)
    
#     # plt.plot(xs, ys, label=name, linewidth=2.7)
#     line = plt.plot(xs, ys, label=name, linewidth=2.6, color=color, linestyle=line_style)[0]
#     # plt.scatter(x, y, marker=marker, color=color, s=50)
#     plt.scatter(x, y, marker=marker, color=color, s=50)
    
#     # Plot the shaded area (confidence interval)
#     plt.fill_between(xs, ys_lower, ys_upper, color=color, alpha=0.2)
    
#     # Annotate points
#     # for xp, yp in zip(x, y):
#     #     plt.annotate(f'{yp:.2f}', (xp, yp), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)
    
#     # Add custom legend entry with both line and marker
#     # legend_handles.append(mlines.Line2D([], [], color=line.get_color(), marker=marker, linestyle=line_style, label=name, linewidth=2.6))
#     legend_handles.append(mlines.Line2D([], [], color=line.get_color(), marker=marker, linestyle=line_style, label=name, linewidth=2.6))

# # Setting the x-axis as described in task
# plt.xticks(x_scales_numbers, x_scales, fontsize=14)
# # setting the font size of y-axis
# plt.yticks(fontsize=14)

# # Create labels and title with larger font size
# plt.xlabel("Input Context Length (# of Words)", fontsize=22)
# plt.ylabel("Accuracy", fontsize=20) # across 4 benchmarks
# # plt.title("SubTask 1 --- Input Context Scaling Trend", fontsize=26)


# # Add grid to the figure, with a light gray color and a dashed line style
# plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# # Show the legend with Times New Roman font (also show the marker)
# # plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper left', markerscale=1.5)
# # Use custom legend handles
# plt.legend(handles=legend_handles, prop={'family': 'Times New Roman', 'size': 15}, loc='upper left')

# # Making the plot layout tight
# plt.tight_layout()

# # Save the plot as a PDF
# plt.savefig("subtask1_input_context_scaling.pdf", format='pdf', bbox_inches='tight')