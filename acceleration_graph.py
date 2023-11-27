import matplotlib.pyplot as plt
from collections import defaultdict

def plot_comparison(benchmarks, your_algorithm_times, normal_execution_times, filename):
    # Create line graph
    plt.figure(figsize=(8, 6))

    plt.plot(benchmarks, your_algorithm_times, marker='o', linestyle='-', label='Our Method')
    plt.plot(benchmarks, normal_execution_times, marker='o', linestyle='-', label='Full Execution')

    plt.xlabel('Benchmarks')
    plt.ylabel('Time Elasped (in seconds)')
    plt.title('Time Taken by Our Method vs Full Execution')
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)
    # plt.yscale('log')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)

data = defaultdict(dict)
data["2D"] = defaultdict(dict)
data["3D"] = defaultdict(dict)

with open("acceleration.txt", "r") as f:
    content = f.read()

for line in content.split('\n'):
    l = line.split(",")
    if len(l) == 3:
        benchmark, type, time = l[0],l[1],l[2]
        if "2D" in benchmark:
            benchmark = benchmark.split('.')[0]
            if type in data["2D"][benchmark]:
                data["2D"][benchmark][type].append(float(time))
            else:
                data["2D"][benchmark][type] = [float(time)]
        elif "3D" in benchmark:
            benchmark = benchmark.split('.')[0]
            if type in data["3D"][benchmark]:
                data["3D"][benchmark][type].append(float(time))
            else:
                data["3D"][benchmark][type] = [float(time)]

   

benchmarks = [key for key in data["2D"].keys()]

hill_climbing_times_2D = [sum(data["2D"][benchmark]["hillClimbing"])/len(data["2D"][benchmark]["hillClimbing"]) for benchmark in benchmarks]
hill_climbing_times_3D = [sum(data["3D"][benchmark]["hillClimbing"])/len(data["3D"][benchmark]["hillClimbing"]) for benchmark in benchmarks]

full_execution_times_2D = [sum(data["2D"][benchmark]["fullExecution"])/len(data["2D"][benchmark]["fullExecution"]) for benchmark in benchmarks]
full_execution_times_3D = [sum(data["3D"][benchmark]["fullExecution"])/len(data["3D"][benchmark]["fullExecution"]) for benchmark in benchmarks]

# Call the function to create and save the plot
plot_comparison(benchmarks, hill_climbing_times_2D, full_execution_times_2D , '2D_Time_Graph.png')
plot_comparison(benchmarks, hill_climbing_times_3D, full_execution_times_3D , '3D_Time_Graph.png')