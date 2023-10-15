# JacoRLDR
Generalizing Robotic Grasping has always been a challenging task due to the difficulty in adapting the grasp action to objects that have different textures, shapes, and weights. Understanding the safest way to lift a never-seen-before object without dropping it requires a high level of coordination and adaptability that is difficult to reproduce using robots. Recently, implementing Reinforcement Learning (RL) algorithms trained with camera inputs represents one of the most promising visual grasping approaches attempting to tackle this challenge. This research tries to implement a 6DoF robotic system in which actions are regulated by an RL stochastic policy trained with status images from a camera attached directly to the end-effector. The model is also trained in simulation using Domain Randomization, which is a valuable approach to deal with the high variability of the environment and to avoid the high costs associated with direct implementation of the system in the real world.

## Proposed Approach
The goal of this research is to create a system that can generalize the task of grasping when different objects, environments, and robotic architectures are present. In order to achieve this level of generalization, this research uses the following components.

### Operational Space Control (OSC) 
The controller used to drive the robot's end-effector position and orientation. It allows to control robots with redundant DOF by avoiding joint limits and minimizing joint torques.
It can resist unforeseeable force perturbations by guaranteeing a prescribed level of impendence. The OSC and the path-planning algorithm of this research were implemented by the [abr-control](https://github.com/abr/abr_control/tree/a248ec56166f01791857a766ac58ee0920c0861c) package.

### SAC 
Soft Actor-Critic (SAC) is an RL algorithm that optimizes a stochastic policy in an off-policy way by leveraging a replay buffer. It uses the entropy hyperparameter to auto-regulate the exploration and exploitation of experiences. Its goal is to maximize both expected reward and the aforementioned entropy, which allows robust learning and effective control in continuous action spaces. In this project, SAC is extended with reward shaping and transfer learning techniques in order to accelerate policy training.

#### Reward Shaping
Dividing our objective into different sub-goals can direct the policy optimization in a more efficient way, effectively reducing the required exploration time. In a grasping task, a robotic arm must:
1. Get close to the target while its fingers are open.
2. Once the end-effector is close enough to the target, the robot's fingers must close in order to grasp the object.
3. Finally, the end-effector must lift the object.

We can then transform these actions into functions expressing individual subgoals, which will be individually maximized by the policy.

Given the total reward function expressed as
$R_{total} = R_{distance} + R_{gripper} + R_{height}$ where
<p align="center">
<img width="407" alt="Screenshot 2023-10-15 at 17 00 01" src="https://github.com/filyps98/JacoRLDR/assets/21342982/00e3f5cb-9caa-40cf-856f-32a3bcf65a42">
</p>
$R_{distance}$ is the reward corresponding to the distance between target and hand, while $R_{gripper}$ is the reward associated with the action of closing the fingers (A corresponds to the fingers amplitude) and $R_{height}$ is the one associated with the difference in height between the target starting and final position.

#### Neural Network structure
SAC is composed of two different types of Networks, that use the first layers of VGG-16 to extract relevant features such as corners and edges:
* Policy Network: the function that models the system. It outputs an action value sampled from a learned Gaussian distribution by taking a state as input. The following image represents the Policy Network structure:

<p align="center">
  <img width="511" alt="image" src="https://github.com/filyps98/JacoRLDR/assets/21342982/6064fa66-309c-48ce-914c-18e5b2bdf6fd">
</p>
* Soft-Q Value Network: it gives a measure of the final reward from a state and an action input. The following image represents the Soft-Q Network structure:


<p align="center">
<img width="575" alt="image" src="https://github.com/filyps98/JacoRLDR/assets/21342982/29dce0e5-b862-48d3-a673-0d535e437f02">
</p>

### First-person camera view 
In this project, the RGB camera is directly mounted on the end-effector, allowing it to move dynamically along with the robot without including its structure in the image frame. Since the input will be independent of the robot structure, the algorithm will remain valid across different architectures, eliminating the need for additional retraining of the networks.
<p align="center">
<img width="317" alt="image" src="https://github.com/filyps98/JacoRLDR/assets/21342982/093d14d3-10b5-4e58-b9d0-b54889fc82f2">
</p>

### Domain Randomization 
Domain randomization is a technique used to randomize the simulation visuals and dynamics to push the RL policy to learn features that are independent from the domain the robot acts in. Domain Randomization in simulation is achieved by modifying the textures and colors of the target objects, the direction and intensity of the lighting, and the dynamics of systems such as mass, friction, and damping.
<p align="center">
<img width="361" alt="image" src="https://github.com/filyps98/JacoRLDR/assets/21342982/77ef3a66-c19e-4601-9f84-6fef2c6b33e5">
</p>

### The Process
First, the camera captures an image of the environment with a resolution of 224x224 pixels. This image is then fed into an RL algorithm model, which processes it to extract a vector of position, orientation, and force grip value. These values are then inputted into the OSC controller, which uses them to determine how to move the robot in the simulation environment to achieve the desired configuration. Once the robot completes the action, another image of the environment is taken, and the process is repeated.

<p align="center">
<img width="558" alt="image" src="https://github.com/filyps98/JacoRLDR/assets/21342982/f6ac1258-8abc-4189-8c35-2c57d09ad091">
</p>

## Results
The following videos represent the robotic simulation and an example of a successful grasping both in first-person and world-view:

![first-person](https://github.com/filyps98/JacoRLDR/assets/21342982/ae1a2ca2-768d-4333-a2d2-68fb71369798)
![world_view](https://github.com/filyps98/JacoRLDR/assets/21342982/3410d18b-e0e2-47ea-ad0e-da689fd32c15)

The following graphs represent the performance across multiple episodes of two different runs having different target entropy values:

<p align="center">
<img width="237" alt="Screenshot 2023-10-15 at 17 23 50" src="https://github.com/filyps98/JacoRLDR/assets/21342982/dd5767f1-556a-4b30-8331-62cc3116983d">
  <img width="238" alt="Screenshot 2023-10-15 at 17 23 03" src="https://github.com/filyps98/JacoRLDR/assets/21342982/9d574186-305e-4e04-a809-391c4aed70f1">
</p>

The results indicate that the available hardware is inadequate to successfully execute the difficult task at hand. The high number of degrees of freedom, and the complex contacts between the target object and end-effector make the calculation of the trajectory computationally expensive and time-consuming. Additionally, since we are dealing with deep reinforcement learning, a high number of image samples is needed for optimization, which further increases the computation time. The inherent challenges in the task pose significant obstacles to the robot’s consistent and reliable object lifting. Nonetheless, it can be verified that in the short amount of time, the optimization was run, the policy learned to avoid negative reward episodes such as the robot collapsing on the object or the floor and to occasionally get close and lift the target object.

## SET-UP

In order to set up the program, follow the instructions contained in the [mujoco-py](https://github.com/openai/mujoco-py) README, and I recommend installing [mujoco 2.1.0](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0). 
Additionally, the [abr-control](https://github.com/abr/abr_control/tree/a248ec56166f01791857a766ac58ee0920c0861c) package needs to be installed, which contains the Jaco2 simulation and the path planning algorithm.

Pytorch is also required, and training on powerful GPU is recommended.

Additionally, the following packages have to be installed:
```python
pip install pydmps
pip install wandb
```

## RUNNING THE PROGRAM
To train the architecture, go into the sac folder directory and type:
```python
python train.py
```
While for testing type:
```python
python test.py
```



