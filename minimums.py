import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import random

class TerrainAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Terrain Analyzer with PSO")

        # GUI elements
        self.canvas = None
        self.image = None
        self.image_path = None

        # Parameters for PSO
        self.num_particles = tk.IntVar(value=100)
        self.num_iterations = tk.IntVar(value=50)

        load_btn = tk.Button(root, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=10, pady=10)

        analyze_btn = tk.Button(root, text="Analyze", command=self.analyze_image)
        analyze_btn.pack(side=tk.LEFT, padx=10, pady=10)

        reset_btn = tk.Button(root, text="Reset", command=self.reset)
        reset_btn.pack(side=tk.LEFT, padx=10, pady=10)

        particle_label = tk.Label(root, text="Swarm Size:")
        particle_label.pack(side=tk.LEFT, padx=5)
        particle_entry = tk.Entry(root, textvariable=self.num_particles, width=5)
        particle_entry.pack(side=tk.LEFT, padx=5)

        iteration_label = tk.Label(root, text="Iterations:")
        iteration_label.pack(side=tk.LEFT, padx=5)
        iteration_entry = tk.Entry(root, textvariable=self.num_iterations, width=5)
        iteration_entry.pack(side=tk.LEFT, padx=5)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.image_path = file_path
        self.image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image()

    def display_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Clear existing canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.image)
        ax.axis('off')

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def analyze_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Interpret the image's terrain based on RGB values
        height_map = self.image[:, :, 0]  # Use the red channel to approximate height

        # PSO parameters
        num_particles = self.num_particles.get()
        num_iterations = self.num_iterations.get()

        # Particle Swarm Optimization
        height, width = height_map.shape
        particles = [
            {
                "position": np.array([random.randint(0, width - 1), random.randint(0, height - 1)]),
                "velocity": np.array([random.uniform(-1, 1), random.uniform(-1, 1)]),
                "best_position": None,
                "best_value": float("inf"),
            }
            for _ in range(num_particles)
        ]

        global_best_positions = []
        global_best_values = []

        # Set up figure for dynamic visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.image)
        particle_plot, = ax.plot([], [], 'mo', markersize=5, label='Particles')  # Purple dots
        ax.legend()

        def update_particles():
            x_coords = [particle["position"][0] for particle in particles]
            y_coords = [particle["position"][1] for particle in particles]
            particle_plot.set_data(x_coords, y_coords)
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.ion()

        for iteration in range(num_iterations):
            for particle in particles:
                x, y = particle["position"].astype(int)
                value = height_map[y, x]

                # Update personal best
                if value < particle["best_value"]:
                    particle["best_value"] = value
                    particle["best_position"] = particle["position"].copy()

            # Identify global bests for this iteration
            sorted_particles = sorted(particles, key=lambda p: p["best_value"])
            if sorted_particles:
                best_particle = sorted_particles[0]
                if best_particle["best_value"] not in global_best_values:
                    global_best_positions.append(best_particle["best_position"])
                    global_best_values.append(best_particle["best_value"])

                # Update particle velocities and positions
                for particle in particles:
                    inertia = 0.5
                    cognitive = random.uniform(0, 1)
                    social = random.uniform(0, 1)
                    particle["velocity"] = (
                        inertia * particle["velocity"]
                        + cognitive * (particle["best_position"] - particle["position"])
                        + social * (best_particle["best_position"] - particle["position"])
                    )
                    particle["velocity"] = particle["velocity"].astype(float)
                    particle["position"] = (particle["position"] + particle["velocity"]).astype(int)
                    particle["position"] = np.clip(particle["position"], [0, 0], [width - 1, height - 1])

            # Update particle visualization after each iteration
            update_particles()
            plt.pause(0.1)  # Pause to visualize each iteration

        plt.ioff()

        # Display results
        self.display_results(global_best_positions)

    def display_results(self, global_best_positions):
        if not global_best_positions:
            messagebox.showinfo("Result", "No local minima found!")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.image)
        for position in global_best_positions:
            x, y = position.astype(int)
            ax.plot(x, y, 'bo', markersize=8, label='Local Minimum')
        ax.legend()
        ax.axis('off')

        # Clear existing canvas and display results
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def reset(self):
        self.image = None
        self.image_path = None
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

if __name__ == "__main__":
    root = tk.Tk()
    app = TerrainAnalyzer(root)
    root.mainloop()

