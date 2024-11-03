
## WindowPaneSimulator

WindowPaneSimulator is an interactive simulation application that visualizes and manages a population of evolving virtual flies (bugs) within a live camera feed. Leveraging computer vision, machine learning, and a graphical user interface, this simulator provides an engaging platform to observe the behaviors, interactions, and evolution of simulated flies in real-time.

## Key Features

Real-Time Simulation: Observe a dynamic population of flies interacting within a live camera feed.

AI-Driven Behavior: Each fly is equipped with a neural network brain, enabling complex behaviors like movement, mating, and energy management.

Enhanced Visuals: Realistic fly representations with segmented bodies, semi-transparent wings with visible veins, and articulated antennae.

Interactive GUI: Control simulation parameters, monitor statistics, and visualize flies with detailed graphics using a user-friendly interface.

Vision Cone: Flies have a simulated vision cone, allowing them to perceive their environment and make informed decisions.

Energy and Reproduction Mechanics: Flies manage their energy levels, seek out resources, and reproduce to sustain and grow the population.

Visual Effects: Natural birth emergence effects with particle spread, ripples, and fading sparkles to enhance realism.

Data Export: Export real-time statistics to CSV for further analysis.

## Installation

Prerequisites

Python 3.7 or higher installed on your system.

A webcam is required for capturing live video feeds or virtual webcam by OBS.

Clone the Repository
 
git clone https://github.com/anttiluode/WindowPaneSimulator.git

cd WindowPaneSimulator

Create a Virtual Environment (Optional but Recommended)

python -m venv venv

Activate the virtual environment:

On Windows:

venv\Scripts\activate

On macOS and Linux:

source venv/bin/activate

pip install -r requirements.txt

requirements.txt:

plaintext
Copy code
tk
opencv-python
numpy
Pillow
torch
einops
matplotlib

Note: Ensure that PyTorch is installed according to your system's specifications. For GPU support, refer to the official PyTorch installation guide.

## Usage

Launch the Application

python app.py

Camera Selection

Upon launching, a dialog will prompt you to select a camera from the available options. If multiple cameras are connected, choose the desired one. You can also test the camera feed before confirming. Additionally, you 

can use OBS Virtual Cam if you wish to stream. To do this:

Use OBS Virtual Cam:

Start an OBS instance and set up a virtual camera.

In another OBS instance, choose the WindowPaneSimulator application window and stream it.

## Main Interface

Video Display: The main canvas displays the live camera feed with simulated flies overlaid.

Controls Panel: Located at the top, allowing you to adjust simulation parameters such as energy decay, speed, and maximum population.

Statistics: Monitor real-time data including current population, generation count, total births, and deaths.

Change Camera: Switch to a different camera feed without restarting the application.

Population Trend Graph: Visualize the population changes over time.

Interacting with the Simulation
Adjust Parameters: Use the sliders to tweak simulation settings on-the-fly.

Monitor Population: Observe how flies interact, reproduce, and manage their energy.

Logging: All events are logged in flies.log for further analysis.

Saving and Loading Populations

Save Population: Click the "Save Population" button to export the current population state to a JSON file.

Load Population: Click the "Load Population" button to import a previously saved population state.

## Exiting the Application

Close the window or select the appropriate exit option. The application ensures all resources are cleaned up properly upon exit.

## Configuration

The simulation is highly customizable through the BugConfig dataclass. Below are the primary configuration parameters:

Vision Settings:

vision_cone_angle: Angle of the fly's vision cone in radians.

vision_cone_length: Length of the vision cone.

vision_width & vision_height: Resolution of the vision input.

Camera Settings:

camera_index: Index of the camera to use.

Movement Settings:

max_speed: Maximum speed of the flies.

turn_speed: Speed at which flies can turn.

momentum: Influences movement smoothness.

random_movement: Degree of random movement.

hover_amplitude: Amplitude of the hover effect.

wing_speed: Speed of wing flapping animation.

Population Settings:

initial_population: Starting number of flies.

max_population: Maximum allowed population.

min_population: Minimum population before triggering repopulation.

Energy Settings:

energy_decay: Rate at which flies lose energy over time.

mating_threshold: Energy level required to attempt reproduction.

initial_energy: Starting energy for each fly.

reproduction_cost: Energy cost for reproducing.

Spacing Settings:

repulsion_distance: Distance at which flies repel each other.

repulsion_force: Strength of the repulsion force.

personal_space: Personal space radius for each fly.

Bug Detection Settings:

bug_visibility_size: Size of bug markers in vision.

bug_detection_range: How far away bugs can see each other.

Visual Effects:

birth_marker_duration: Duration (in frames) to show birth effects.

death_marker_duration: Duration (in frames) to show death effects.

Genetic Settings:

mutation_rate: Probability of mutation per trait during reproduction.

These parameters can be adjusted via the GUI sliders or by modifying the BugConfig class directly in the source code.


## Architecture

Key Components

BugConfig: Defines the configuration parameters for the simulation, including vision, movement, population, and energy settings.

FlyVisuals: Handles the rendering of flies on the video feed, including body segmentation, wings, antennae, and energy bars. It also visualizes the vision cone of each fly and manages visual effects like birth and death animations.

EnhancedBugBrain (Neural Network): A PyTorch-based neural network that processes visual input from the fly's vision cone to make decisions regarding movement, eating, and mating.

EnhancedBug: Represents each fly in the simulation. Manages the fly's state, energy levels, movement, and interactions with other flies.

BugGUI: The main graphical user interface built with Tkinter. It manages the camera feed, displays the simulation, provides controls for adjusting parameters, shows real-time statistics, and handles data export.

VisualEffect: Represents visual effects such as birth and death animations, managing their lifecycle and rendering on the video feed.

Main Function: Initializes logging, sets up the GUI, and starts the main application loop.

## Workflow

Initialization:

The application starts by selecting and initializing the camera.

An initial population of EnhancedBug instances is created.

## Simulation Loop:

Frames are captured from the camera in a separate thread to ensure smooth performance.

Each frame is processed:

Flies perceive their environment through their vision cones.

The neural network processes visual input to decide on actions.

Flies update their positions, manage energy, and interact (e.g., mating).

Dead flies are removed, and new flies are added through reproduction.

Visual effects are rendered for events like births and deaths.

The GUI is updated with the latest frame and fly visuals.

User Interaction:

Users can adjust simulation parameters in real-time.

Statistics are continuously updated to reflect the current state of the simulation.

Users can save and load population states for persistence and analysis.

## Acknowledgements

Antorphic / OpenAI: For the foundational code and support.

Tkinter: For the graphical user interface components.

OpenCV: For real-time computer vision processing.

PyTorch: For building and training the neural network models.

Pillow: For image processing tasks.

Einops: For tensor manipulation in the neural network.

NumPy: For numerical operations and data handling.

Matplotlib: For data visualization and plotting.

Troubleshooting

Common Errors and Solutions

Camera Initialization Error:

## Logging

All significant events, errors, and warnings are logged in the flies.log file located in the application directory. Reviewing this log can provide insights into the simulation's behavior and help diagnose issues.

## Future Enhancements

While WindowPaneSimulator offers a robust simulation experience, there are always opportunities for improvement. Here are some potential future enhancements:

Dynamic Lighting Effects: Implement day-night cycles or moving light sources to affect fly behaviors.

Interactive Environment: Allow users to add obstacles or food sources within the simulation space.

Advanced Neural Network Training: Incorporate reinforcement learning for flies to adapt and evolve more sophisticated behaviors over time.

Enhanced Motion Physics: Simulate realistic inertia, acceleration, and collision responses for flies.

State-Based Visual Indicators: Use visual cues to represent different states or moods of flies, such as hunger or fatigue.

## Licence

MIT Licence 
