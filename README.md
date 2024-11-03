# WindowPaneSimulator
WindowPaneSimulator is an interactive simulation application that visualizes and manages a population of evolving virtual flies (bugs) within a live camera feed. Leveraging computer vision, machine learning, and a graphical user interface, this simulator provides an engaging platform to observe the behaviors, interactions, and evolution of simulated flies in real-time.

Real-Time Simulation: Observe a dynamic population of flies interacting within a live camera feed.

AI-Driven Behavior: Each fly is equipped with a neural network brain, enabling complex behaviors like movement, mating, and energy management.

Interactive GUI: Control simulation parameters, monitor statistics, and visualize flies with detailed graphics using a user-friendly interface.

Vision Cone: Flies have a simulated vision cone, allowing them to perceive their environment and make informed decisions.

Energy and Reproduction Mechanics: Flies manage their energy levels, seek out resources, and reproduce to sustain and grow the population.

# Installation

Prerequisites

Ensure you have Python 3.7 or higher installed on your system. Additionally, a webcam is required for capturing live video feeds.

Clone the Repository

git clone https://github.com/anttiluode/WindowPaneSimulator.git

cd WindowPaneSimulator

Create a Virtual Environment (Optional but Recommended)

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

requirements.txt

tk
opencv-python
messagebox
numpy
Pillow
torch
einops

Note: Ensure that PyTorch is installed according to your system's specifications. For GPU support, refer to the official PyTorch installation guide.

Usage
Launch the Application

python app.py

# Camera Selection

Upon launching, a dialog will prompt you to select a camera from the available options. If multiple cameras are connected, choose the desired one. You can also test the camera feed before confirming. You could also use OBS virtual cam, if you want to stream. Then have another OBS instance where 
you can choose the app window and stream that. 

# Main Interface

Video Display: The main canvas displays the live camera feed with simulated flies overlaid.

Controls Panel: Located at the top, allowing you to adjust simulation parameters such as energy decay, speed, and maximum population.

Statistics: Monitor real-time data including current population, generation count, total births, and deaths.

Change Camera: Switch to a different camera feed without restarting the application.
Interacting with the Simulation

Adjust Parameters: Use the sliders to tweak simulation settings on-the-fly.
Monitor Population: Observe how flies interact, reproduce, and manage their energy.
Logging: All events are logged in flies.log for further analysis.
Exiting the Application

Close the window or select the appropriate exit option. The application ensures all resources are cleaned up properly upon exit.

# Configuration
The simulation is highly customizable through the BugConfig dataclass. Below are the primary configuration parameters:

# Vision Settings:

vision_cone_angle: Angle of the fly's vision cone in radians.
vision_cone_length: Length of the vision cone.
vision_width & vision_height: Resolution of the vision input.

# Camera Settings:

camera_index: Index of the camera to use.
 
# Movement Settings:

max_speed: Maximum speed of the flies.
turn_speed: Speed at which flies can turn.
momentum: Influences movement smoothness.
random_movement: Degree of random movement.
hover_amplitude: Amplitude of the hover effect.
wing_speed: Speed of wing flapping animation.

# Population Settings:

initial_population: Starting number of flies.
max_population: Maximum allowed population.
min_population: Minimum population before triggering repopulation.
Energy Settings:

energy_decay: Rate at which flies lose energy over time.
mating_threshold: Energy level required to attempt reproduction.
initial_energy: Starting energy for each fly.
reproduction_cost: Energy cost for reproducing.
These parameters can be adjusted via the GUI sliders or by modifying the BugConfig class directly in the source code.

# Architecture

Key Components

BugConfig: Defines the configuration parameters for the simulation, including vision, movement, population, and energy settings.

FlyVisuals: Handles the rendering of flies on the video feed, including body, wings, antennae, and energy bars. It also visualizes the vision cone of each fly.

EnhancedBugBrain (Neural Network): A PyTorch-based neural network that processes visual input from the fly's vision cone to make decisions regarding movement, eating, and mating.

EnhancedBug: Represents each fly in the simulation. Manages the fly's state, energy levels, movement, and interactions with other flies.

BugGUI: The main graphical user interface built with tkinter. It manages the camera feed, displays the simulation, provides controls for adjusting parameters, and shows real-time statistics.

Main Function: Initializes logging, sets up the GUI, and starts the main application loop.

# Workflow

Initialization:

The application starts by selecting and initializing the camera.
An initial population of EnhancedBug instances is created.

Simulation Loop:

Frames are captured from the camera in a separate thread to ensure smooth performance.

Each frame is processed:

Flies perceive their environment through their vision cones.
The neural network processes visual input to decide on actions.
Flies update their positions, manage energy, and interact (e.g., mating).
Dead flies are removed, and new flies are added through reproduction.
The GUI is updated with the latest frame and fly visuals.

# User Interaction:

Users can adjust simulation parameters in real-time.
Statistics are continuously updated to reflect the current state of the simulation.
Contributing
Contributions are welcome! Whether it's reporting bugs, suggesting features, or submitting pull requests, your input helps improve the WindowPaneSimulator.

# License
This project is licensed under the MIT License.

Acknowledgements

Antorphic / OpenAI for coding it. 
Tkinter: For the graphical user interface components.
OpenCV: For real-time computer vision processing.
PyTorch: For building and training the neural network models.
Pillow: For image processing tasks.
Einops: For tensor manipulation in the neural network.
NumPy: For numerical operations and data handling.
