import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
from dataclasses import dataclass
from collections import deque
import logging
import sys
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange

@dataclass
@dataclass
class BugConfig:
    # Vision settings
    vision_cone_angle: float = np.pi/3  # 60 degrees in radians
    vision_cone_length: int = 150
    vision_width: int = 32
    vision_height: int = 32
    
    # Camera settings
    camera_index: int = 0       # Default camera index
    
    # Movement settings
    max_speed: float = 15.0     # Increased for faster movement
    turn_speed: float = 0.3     # Increased for quicker turns
    momentum: float = 0.7       # Reduced for more erratic movement
    random_movement: float = 0.2 # Random movement factor
    hover_amplitude: float = 2.0 # Hover movement
    wing_speed: float = 0.5     # Wing flapping speed
    
    # Population settings
    initial_population: int = 15
    max_population: int = 100   # Increased max population
    min_population: int = 5     # Minimum population

    # Energy settings
    energy_decay: float = 0.05  # Reduced for longer life
    mating_threshold: float = 80.0
    initial_energy: float = 100.0
    reproduction_cost: float = 40.0

class FlyVisuals:
    """Handle fly drawing with wings and antennae"""
    def __init__(self, base_size=10):
        self.base_size = base_size
        self.wing_phase = 0.0
    
    def draw_fly(self, frame, x, y, angle, is_mating, energy_ratio):
        # Convert coordinates to integers
        x, y = int(x), int(y)
        
        # Update wing animation
        self.wing_phase += 0.3
        wing_offset = np.sin(self.wing_phase) * self.base_size

        # Body points (oval shape)
        body_length = self.base_size * 2
        body_width = self.base_size
        
        # Calculate body points
        body_points = np.array([
            [x - body_length * np.cos(angle), y - body_length * np.sin(angle)],
            [x + body_length * np.cos(angle), y + body_length * np.sin(angle)]
        ], dtype=np.int32)

        # Draw body
        color = (200, 50, 200) if is_mating else (40, 40, 40)  # Dark gray for normal, purple for mating
        cv2.line(frame, tuple(body_points[0]), tuple(body_points[1]), color, thickness=body_width)

        # Draw wings (animated)
        wing_angle1 = angle + np.pi/2 + np.sin(self.wing_phase) * 0.5
        wing_angle2 = angle - np.pi/2 - np.sin(self.wing_phase) * 0.5
        
        wing_length = self.base_size * 1.5
        
        wing1_end = (
            int(x + wing_length * np.cos(wing_angle1)),
            int(y + wing_length * np.sin(wing_angle1))
        )
        wing2_end = (
            int(x + wing_length * np.cos(wing_angle2)),
            int(y + wing_length * np.sin(wing_angle2))
        )
        
        # Semi-transparent wings
        wing_color = (200, 200, 200, 150)  # Light gray, semi-transparent
        cv2.line(frame, (x, y), wing1_end, wing_color, thickness=2)
        cv2.line(frame, (x, y), wing2_end, wing_color, thickness=2)

        # Draw antennae
        antenna_length = self.base_size * 0.8
        antenna_angle1 = angle + np.pi/4
        antenna_angle2 = angle - np.pi/4
        
        ant1_end = (
            int(x + antenna_length * np.cos(antenna_angle1)),
            int(y + antenna_length * np.sin(antenna_angle1))
        )
        ant2_end = (
            int(x + antenna_length * np.cos(antenna_angle2)),
            int(y + antenna_length * np.sin(antenna_angle2))
        )
        
        cv2.line(frame, (x, y), ant1_end, color, thickness=1)
        cv2.line(frame, (x, y), ant2_end, color, thickness=1)

        # Draw energy bar
        bar_length = 20
        energy_width = int(energy_ratio * bar_length)
        energy_color = (0, 255, 0) if energy_ratio > 0.5 else (255, 165, 0)
        cv2.rectangle(
            frame,
            (x - bar_length//2, y - self.base_size * 3),
            (x - bar_length//2 + energy_width, y - self.base_size * 3 + 2),
            energy_color, -1
        )

    def draw_vision_cone(self, frame, bug, config):
        """Draw the vision cone with gradient"""
        angle = config.vision_cone_angle
        length = config.vision_cone_length
        
        # Calculate cone points
        left_angle = bug.angle - angle/2
        right_angle = bug.angle + angle/2
        
        points = np.array([
            [int(bug.x), int(bug.y)],
            [int(bug.x + length * np.cos(left_angle)), 
             int(bug.y + length * np.sin(left_angle))],
            [int(bug.x + length * np.cos(bug.angle)),
             int(bug.y + length * np.sin(bug.angle))],
            [int(bug.x + length * np.cos(right_angle)), 
             int(bug.y + length * np.sin(right_angle))]
        ], dtype=np.int32)

        # Create gradient for vision cone
        if bug.is_mating:
            color1 = (200, 50, 200, 30)  # Purple, very transparent
            color2 = (200, 50, 200, 5)   # Purple, mostly transparent
        else:
            color1 = (50, 200, 50, 30)   # Green, very transparent
            color2 = (50, 200, 50, 5)    # Green, mostly transparent
            
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [points], True, color2, 1)
class EnhancedBugBrain(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        
        # Vision processing
        self.vision_net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # Reduce spatial dimensions
        )
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim * 64, hidden_dim),  # 8x8 = 64 spatial locations
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decision network
        self.decision_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # velocity, rotation, eating, mating
        )
        
        # Memory buffer
        self.memory_size = 16
        self.memory = deque(maxlen=self.memory_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, vision_input):
        # Ensure input is the right shape and type
        if not isinstance(vision_input, torch.Tensor):
            vision_input = torch.FloatTensor(vision_input)
        
        # Add batch dimension if needed
        if len(vision_input.shape) == 3:
            vision_input = vision_input.unsqueeze(0)
        
        # Normalize input
        vision_input = vision_input / 255.0
        
        # Process vision
        visual_features = self.vision_net(vision_input)
        
        # Flatten spatial dimensions
        visual_features = visual_features.flatten(1)
        
        # Process features
        features = self.feature_net(visual_features)
        
        # Store in memory
        self.memory.append(features.detach())
        
        # Get decisions
        decisions = self.decision_net(features)
        
        # Split decisions
        velocity = torch.tanh(decisions[:, 0])
        rotation = torch.tanh(decisions[:, 1])
        eating = torch.sigmoid(decisions[:, 2])
        mating = torch.sigmoid(decisions[:, 3])
        
        return velocity, rotation, eating.item(), mating.item()

class EnhancedBug:
    def __init__(self, x: int, y: int, config: BugConfig):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * np.pi)
        self.config = config
        
        # Initialize brain
        self.brain = EnhancedBugBrain()
        
        # State
        self.energy = config.initial_energy
        self.age = 0
        self.mating_cooldown = 0
        self.is_mating = False
        self.memory = deque(maxlen=10)
        
        # Movement state
        self.current_velocity = 0.0
        self.current_rotation = 0.0
        
        # Enhanced movement
        self.hover_offset = random.uniform(0, 2 * np.pi)
        self.hover_time = 0
        self.random_direction = random.uniform(-1, 1)
        self.last_direction_change = 0
    
    def get_vision_cone(self, frame: np.ndarray) -> np.ndarray:
        """Extract the portion of the frame visible in the bug's vision cone"""
        height, width = frame.shape[:2]
        
        # Create mask for vision cone
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate vision cone points
        angle = self.config.vision_cone_angle
        length = self.config.vision_cone_length
        
        # Calculate cone edges
        left_angle = self.angle - angle/2
        right_angle = self.angle + angle/2
        
        # Create polygon points for cone
        points = np.array([
            [int(self.x), int(self.y)],
            [int(self.x + length * np.cos(left_angle)), 
             int(self.y + length * np.sin(left_angle))],
            [int(self.x + length * np.cos(right_angle)), 
             int(self.y + length * np.sin(right_angle))]
        ], dtype=np.int32)
        
        # Draw filled cone on mask
        cv2.fillPoly(mask, [points], 255)
        
        # Apply mask to frame
        cone_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Resize to desired resolution
        cone_image = cv2.resize(cone_image, 
                              (self.config.vision_width, self.config.vision_height))
        
        return cone_image

    def process_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """Process frame and decide movement"""
        try:
            # Extract vision cone
            cone_image = self.get_vision_cone(frame)
            
            # Convert for PyTorch (C, H, W)
            cone_tensor = torch.FloatTensor(cone_image).permute(2, 0, 1)
            
            # Get brain's decisions
            velocity, rotation, eating, mating = self.brain(cone_tensor)
            
            # Update state
            self.is_mating = mating > 0.5
            
            # Update energy
            self._update_energy(cone_image, eating > 0.5)
            
            # Apply momentum
            self.current_velocity = (self.current_velocity * self.config.momentum + 
                                   velocity.item() * self.config.max_speed * (1 - self.config.momentum))
            self.current_rotation = (self.current_rotation * self.config.momentum + 
                                   rotation.item() * self.config.turn_speed * (1 - self.config.momentum))
            
            return self.current_velocity, self.current_rotation
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return 0.0, 0.0
    
    def _update_energy(self, vision_cone: np.ndarray, want_eat: bool):
        """Update bug's energy based on actions and environment"""
        # Base energy decay
        self.energy -= self.config.energy_decay
        
        if want_eat:
            # Calculate energy from light levels
            light_level = np.mean(vision_cone) / 255.0
            energy_gain = light_level * 2.0
            
            # Bonus energy from movement in the frame
            if len(self.memory) > 0:
                prev_frame = self.memory[-1]
                movement = np.mean(np.abs(vision_cone - prev_frame))
                energy_gain += movement * 5.0
            
            self.energy += energy_gain
            
        self.memory.append(vision_cone.copy())
    
    def try_mate(self, other: 'EnhancedBug') -> Optional['EnhancedBug']:
        """Attempt to mate with another bug"""
        if (self.is_mating and other.is_mating and 
            self.energy > self.config.mating_threshold and 
            other.energy > self.config.mating_threshold and
            self.mating_cooldown == 0 and other.mating_cooldown == 0):
            
            # Create child
            child = EnhancedBug(
                x=(self.x + other.x) / 2,
                y=(self.y + other.y) / 2,
                config=self.config
            )
            
            # Update parent states
            self.energy -= self.config.reproduction_cost
            other.energy -= self.config.reproduction_cost
            self.mating_cooldown = 100
            other.mating_cooldown = 100
            
            return child
        
        return None

    def update_position(self, velocity: float, rotation: float):
        """Update position with enhanced movement"""
        # Update hover
        self.hover_time += 0.1
        hover_y = np.sin(self.hover_time + self.hover_offset) * self.config.hover_amplitude
        
        # Random direction changes
        if self.age - self.last_direction_change > 30:
            if random.random() < self.config.random_movement:
                self.random_direction = random.uniform(-1, 1)
                self.last_direction_change = self.age
        
        # Add random movement to rotation
        rotation += self.random_direction * self.config.turn_speed
        
        # Update angle with enhanced rotation
        self.angle += rotation
        
        # Calculate new position with hover effect
        self.x += velocity * np.cos(self.angle)
        self.y += velocity * np.sin(self.angle) + hover_y
        
        # Keep bug on screen by wrapping around edges
        if hasattr(self, 'screen_width'):
            self.x = self.x % self.screen_width
            self.y = self.y % self.screen_height
        
        # Update state
        self.age += 1
        if self.mating_cooldown > 0:
            self.mating_cooldown -= 1
            
class BugGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Evolving Flies Simulation")
        
        # Create config and setup controls first
        self.config = BugConfig()
        self.setup_camera_selection()
        
        # Initialize camera later
        self.initialize_camera()
            
        # Create bugs
        self.bugs: List[EnhancedBug] = []
        self.fly_visuals = FlyVisuals()
        self.initialize_population()
        
        # Setup rest of GUI elements
        self.setup_gui()
        
        # Processing queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        
        # Statistics
        self.generation = 1
        self.total_births = 0
        self.total_deaths = 0
        
        # Start processing
        self.start_processing()

    def start_processing(self):
        """Start processing threads"""
        self.running = True
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Start processing loop
        self.process_frame()
    
    def initialize_population(self):
        """Create initial bug population"""
        for _ in range(self.config.initial_population):
            bug = EnhancedBug(
                x=random.randint(0, self.width),
                y=random.randint(0, self.height),
                config=self.config
            )
            bug.screen_width = self.width
            bug.screen_height = self.height
            self.bugs.append(bug)

    def setup_camera_selection(self):
        """Create camera selection dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Camera Selection")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select Camera:").pack(padx=10, pady=5)
        
        # Create camera list
        camera_frame = ttk.Frame(dialog)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Try first 5 camera indices
        self.available_cameras = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.available_cameras.append(i)
                    cap.release()
            except:
                continue
        
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras found!")
            sys.exit(1)
        
        # Camera selection dropdown
        self.camera_var = tk.IntVar(value=self.available_cameras[0])
        camera_select = ttk.Combobox(
            camera_frame, 
            textvariable=self.camera_var,
            values=self.available_cameras,
            state='readonly',
            width=10
        )
        camera_select.pack(side=tk.LEFT, padx=5)
        
        # Test button
        def test_camera():
            try:
                cap = cv2.VideoCapture(self.camera_var.get())
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Show small preview
                        frame = cv2.resize(frame, (320, 240))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        photo = ImageTk.PhotoImage(image)
                        
                        preview = tk.Toplevel(dialog)
                        preview.title("Camera Preview")
                        label = ttk.Label(preview, image=photo)
                        label.image = photo
                        label.pack()
                        
                        def close_preview():
                            cap.release()
                            preview.destroy()
                        
                        preview.after(2000, close_preview)  # Close after 2 seconds
                    cap.release()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to test camera: {str(e)}")
        
        ttk.Button(camera_frame, text="Test", command=test_camera).pack(side=tk.LEFT, padx=5)    

        
        # OK button
        def on_ok():
            self.config.camera_index = self.camera_var.get()
            dialog.destroy()
            
        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        
        # Wait for dialog
        self.root.wait_window(dialog)

    def initialize_camera(self):
        """Initialize selected camera"""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Could not open camera {self.config.camera_index}")
            
            # Set desired resolution - adjust these values as needed
            desired_width = 1280  # or 640, 800, 1920, etc.
            desired_height = 720  # or 480, 600, 1080, etc.
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
                
            # Get actual resolution (might be different from requested)
            self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Camera initialized at resolution: {self.width}x{self.height}")
                
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to initialize camera: {str(e)}")
            sys.exit(1)

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel at top
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left side controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(left_controls, text="Statistics", padding="5")
        stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.population_var = tk.StringVar(value="Population: 0")
        self.generation_var = tk.StringVar(value="Generation: 1")
        self.births_var = tk.StringVar(value="Births: 0")
        self.deaths_var = tk.StringVar(value="Deaths: 0")
        
        ttk.Label(stats_frame, textvariable=self.population_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.generation_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.births_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.deaths_var).pack(side=tk.LEFT, padx=5)
        
        # Camera selection button
        def change_camera():
            self.cleanup()
            self.setup_camera_selection()
            self.initialize_camera()
            self.start_processing()
            
        ttk.Button(stats_frame, text="Change Camera", command=change_camera).pack(side=tk.LEFT, padx=5)
        
        # Right side controls
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT, fill=tk.X)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(right_controls, text="Parameters", padding="5")
        params_frame.pack(side=tk.RIGHT, fill=tk.X, padx=5)

        # Energy decay slider
        ttk.Label(params_frame, text="Energy:").pack(side=tk.LEFT)
        self.energy_scale = ttk.Scale(
            params_frame, 
            from_=0.01, 
            to=0.2,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.config.energy_decay
        )
        self.energy_scale.pack(side=tk.LEFT, padx=5)

        # Speed slider
        ttk.Label(params_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_scale = ttk.Scale(
            params_frame, 
            from_=5.0, 
            to=25.0,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.config.max_speed
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Population slider
        ttk.Label(params_frame, text="Max Pop:").pack(side=tk.LEFT)
        self.pop_scale = ttk.Scale(
            params_frame, 
            from_=10, 
            to=200,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.config.max_population
        )
        self.pop_scale.pack(side=tk.LEFT, padx=5)
        
        # Canvas for video display
        self.canvas = tk.Canvas(main_frame, width=self.width, height=self.height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def camera_loop(self):
        """Background thread for camera capture"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(1/30)  # Limit to 30fps
    
    def process_frame(self):
        """Main processing loop"""
        try:
            if not self.frame_queue.empty():
                # Get frame from queue
                frame = self.frame_queue.get()
                
                # Update parameters from sliders
                self.config.energy_decay = self.energy_scale.get()
                self.config.max_speed = self.speed_scale.get()
                
                # Process all bugs
                dead_bugs = []
                new_bugs = []
                
                # Update bugs
                for bug in self.bugs:
                    # Process through bug's brain
                    velocity, rotation = bug.process_frame(frame)
                    
                    # Update bug position
                    bug.update_position(velocity, rotation)
                    
                    # Check for death
                    if bug.energy <= 0:
                        dead_bugs.append(bug)
                        self.total_deaths += 1
                    
                    # Check for reproduction
                    if bug.is_mating and len(self.bugs) < self.config.max_population:
                        for other in self.bugs:
                            if other != bug:
                                child = bug.try_mate(other)
                                if child:
                                    child.screen_width = self.width
                                    child.screen_height = self.height
                                    new_bugs.append(child)
                                    self.total_births += 1
                                    break
                
                # Remove dead bugs
                for bug in dead_bugs:
                    self.bugs.remove(bug)
                
                # Add new bugs
                self.bugs.extend(new_bugs)
                
                # Check population thresholds
                if len(self.bugs) < self.config.min_population:
                    self.generation += 1
                    self.repopulate()
                
                # Update display
                self.update_display(frame)
                
                # Update statistics
                self.update_stats()
                
            # Schedule next update
            self.root.after(33, self.process_frame)
            
        except Exception as e:
            logging.error(f"Processing error: {e}")
            messagebox.showerror("Error", str(e))
    
    def update_stats(self):
        """Update statistics display"""
        self.population_var.set(f"Population: {len(self.bugs)}")
        self.generation_var.set(f"Generation: {self.generation}")
        self.births_var.set(f"Births: {self.total_births}")
        self.deaths_var.set(f"Deaths: {self.total_deaths}")

    def repopulate(self):
        """Add new bugs when population is too low"""
        while len(self.bugs) < self.config.initial_population:
            bug = EnhancedBug(
                x=random.randint(0, self.width),
                y=random.randint(0, self.height),
                config=self.config
            )
            bug.screen_width = self.width
            bug.screen_height = self.height
            self.bugs.append(bug)

    def update_display(self, frame):
        """Update display with flies only"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw all bugs
        for bug in self.bugs:
            # Draw fly with full visuals
            self.fly_visuals.draw_fly(
                frame_rgb,
                bug.x,
                bug.y,
                bug.angle,
                bug.is_mating,
                bug.energy / self.config.initial_energy
            )
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.photo = photo

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'camera_thread') and self.camera_thread is not None:
            self.camera_thread.join()
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("flies.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create main window
    root = tk.Tk()
    
    try:
        # Create and run application
        app = BugGUI(root)
        
        # Setup cleanup
        def on_closing():
            app.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()