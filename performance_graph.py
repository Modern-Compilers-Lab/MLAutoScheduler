import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Function to generate speedup dictionary for given tile dimension (2D or 3D)
def generate_speedup(tile_dimension):
    speedup = {
        'No Tiling': tuple(perf_data[f"{benchmark}.{tile_dimension}"]["NoTiling"] for benchmark in benchmarks),
        'Tiling with Default Parameters': tuple(perf_data[f"{benchmark}.{tile_dimension}"]["DefaultParameters"] for benchmark in benchmarks),
        'Tiling with Our Method': tuple(perf_data[f"{benchmark}.{tile_dimension}"]["BestParameters"] for benchmark in benchmarks),
    }
    return speedup

# Data reading and processing.
perf_data = defaultdict(dict)
with open("performanceResults.txt", "r") as f:
    for line in f:
        content = line.strip().split(",")
        if len(content) > 3:  # For lines with more than 3 elements
            benchmark = content[0]
            label = content[1]
            tilesize = ",".join(content[2:])  # Combine remaining elements into tilesize
            if benchmark not in perf_data:
                perf_data[benchmark] = defaultdict(list)
            perf_data[benchmark][label] = tilesize
        elif len(content) == 3:  # For lines with exactly 3 elements
            benchmark, label = content[0], content[1]
            speed = float(content[2].split()[0])  # Extract speed as a float
            if benchmark not in perf_data:
                perf_data[benchmark] = defaultdict(list)
            perf_data[benchmark][label].append(speed)  # Store speeds in a list


# Loop through the accumulated lists in perf_data and calculate averages
for benchmark, labels_data in perf_data.items():
    for label, speed_list in labels_data.items():
        if isinstance(speed_list, list) and speed_list:  # Check if it's a non-empty list
            average_speed = sum(speed_list) / len(speed_list)  # Calculate average speed
            perf_data[benchmark][label] = average_speed  # Store average as a float

# Define benchmarks
benchmarks = ("conv1d", "conv2d", "conv3d", "matmul")

# Generate speedup dictionaries for 2D and 3D
speedup_2D = generate_speedup("mlir2D")
speedup_3D = generate_speedup("mlir3D")

x = np.arange(len(benchmarks))  # the label locations
width = 0.30  # the width of the bars

# Define a function to handle plotting for 2D and 3D speedup
def plot_speedup(ax, speedup_data, title):
    multiplier = 0
    for attribute, measurement in speedup_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        for rect, value in zip(rects, measurement):  # Iterate through bars and values
            if value != 0 and abs(value) < 0.0001:  # For non-zero very small values, represent in scientific notation
                height = rect.get_height()
                ax.annotate('{:.2e}'.format(value),  # Scientific notation format
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 6),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)
            else:  # For other values including zero, annotate normally
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

        multiplier += 1

    # Customize the plot
    ax.set_ylabel('Speedup')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(speedup_data) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    ax.set_yscale('log')

# Create and save figures for 2D and 3D speedup comparison separately
fig, ax1 = plt.subplots(figsize=(10, 6))  # Figure for 2D Speedup
plot_speedup(ax1, speedup_2D, '2D Speedup Comparison')
plt.tight_layout()
plt.savefig('2D_Speedup_Comparison.png')
plt.close()

fig, ax2 = plt.subplots(figsize=(10, 6))  # Figure for 3D Speedup
plot_speedup(ax2, speedup_3D, '3D Speedup Comparison')
plt.tight_layout()
plt.savefig('3D_Speedup_Comparison.png')
plt.close()
