# The Solution

This folder contains the soltution that was implemented. Our solution included a custom implementation of Graph SLAM and VPR. The files player_autonomous_PoseGraph.py contains the source code.

# Implementation
## Graph SLAM
Grph SLAM stores the keypoints and descriptors alongside the current pose as a linked list. This helps us track the flow/movement of the robot through the environment. The pose is comuted with forward stepping the keystroke. This linked list forms our Graph SLAM base. Loop closure is not considered in the solution, but adding a loop closure with multiple pointers to this solution will make a better representation of the map.   

## Visual Place Recognition
The VPR in this solution is implemented as a simple key point and feature search over the Graph. The node with maximum match over a threshold is then extracted as the target location. This solution can be further fine tuned with an implementation leveraging Visual Bag Of Words or VLAD leveraging the fact that the patterns that could occur in the maze are already know.

Below is a visual representation of the solution.
Graph SLAM:
![Graph SLAM: Building the Graph](/Solution/GraphSLAM.gif)

VPR and Navigation:
![Graph SLAM: Building the Graph](/Solution/VPR-gif.gif)

