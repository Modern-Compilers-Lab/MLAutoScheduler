#!/bin/bash

# Change directory to 'build'
cd /Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/build || exit

# Clear the content of performance.txt and acceleration.txt (if it exists) or create an empty file
> /Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/performanceResults.txt
> /Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/acceleration.txt

# Loop to run the commands 5 times
for i in {1..5}
do
    for operation in conv1d conv2d conv3d matmul
    do
        cmake --build . && bin/AutoSchedulerML ../benchmarks/"$operation".mlir 2D 3D
    done
done

# run perf.py and acceleration_graph.py
python3 /Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/performance_graph.py
python3 /Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/acceleration_graph.py