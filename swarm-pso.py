import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor with PSO")
        
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.image = None
        self.image_tk = None
        self.drawings = []  # List of drawn items
        self.swarm = []  # Swarm particles
        
        # Buttons and inputs
        self.load_button = tk.Button(self.toolbar, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_button = tk.Button(self.toolbar, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Label(self.toolbar, text="Swarm Size:").pack(side=tk.LEFT, padx=5)
        self.swarm_size_entry = tk.Entry(self.toolbar, width=5)
        self.swarm_size_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.toolbar, text="Iterations:").pack(side=tk.LEFT, padx=5)
        self.iterations_entry = tk.Entry(self.toolbar, width=5)
        self.iterations_entry.pack(side=tk.LEFT, padx=5)

        self.algorithm_label = tk.Label(self.toolbar, text="Algorithm:")
        self.algorithm_label.pack(side=tk.LEFT, padx=5)

        self.algorithm_selector = ttk.Combobox(self.toolbar, values=["PSO"], state="readonly")
        self.algorithm_selector.set("PSO")
        self.algorithm_selector.pack(side=tk.LEFT, padx=5)

        self.run_button = tk.Button(self.toolbar, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Event bindings
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)

        # PSO settings
        self.particles = []
        self.current_line = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.drawings.clear()
            self.swarm.clear()

    def start_drawing(self, event):
        x, y = event.x, event.y
        self.current_line = [(x, y)]
        point = self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="red", outline="red")
        self.drawings.append(("point", (x, y)))

    def draw_line(self, event):
        if self.current_line:
            x, y = event.x, event.y
            x1, y1 = self.current_line[-1]
            line = self.canvas.create_line(x1, y1, x, y, fill="red")
            self.current_line.append((x, y))

    def end_drawing(self, event):
        if self.current_line and len(self.current_line) > 1:
            self.drawings.append(("line", self.current_line))
        self.current_line = None

    def clear_drawing(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)
        self.drawings.clear()
        self.swarm.clear()

    def calculate_fitness(self, particle_pos):
        fitness = 0
        for shape, coords in self.drawings:
            if shape == "point":
                px, py = coords
                fitness += np.linalg.norm(np.array(particle_pos) - np.array([px, py]))
            elif shape == "line":
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    # Distance from point to line segment
                    line_vec = np.array([x2 - x1, y2 - y1])
                    point_vec = np.array(particle_pos) - np.array([x1, y1])
                    line_len = np.linalg.norm(line_vec)
                    if line_len > 0:
                        line_unit_vec = line_vec / line_len
                        proj_length = np.dot(point_vec, line_unit_vec)
                        proj_length = max(0, min(line_len, proj_length))
                        closest_point = np.array([x1, y1]) + proj_length * line_unit_vec
                        fitness += np.linalg.norm(particle_pos - closest_point)
        return fitness

    def run_simulation(self):
        try:
            swarm_size = int(self.swarm_size_entry.get())
            iterations = int(self.iterations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Swarm Size and Iterations must be integers.")
            return

        algorithm = self.algorithm_selector.get()

        if algorithm == "PSO":
            self.run_pso(swarm_size, iterations)
        else:
            messagebox.showerror("Error", "Algorithm not implemented.")

    def run_pso(self, swarm_size, iterations):
        if not self.image:
            messagebox.showerror("Error", "Please load an image before running the simulation.")
            return

        # Initialize particles
        self.particles = [
            {
                "position": np.array([
                    np.random.randint(0, self.image.width),
                    np.random.randint(0, self.image.height)
                ], dtype=np.float64),
                "velocity": np.random.uniform(-1, 1, 2),
                "best_position": None,
                "best_value": float("inf")
            }
            for _ in range(swarm_size)
        ]

        global_best_position = None
        global_best_value = float("inf")

        for iteration in range(iterations):
            for particle in self.particles:
                # Calculate fitness based on user-drawn points and lines
                fitness = self.calculate_fitness(particle["position"])
                if fitness < particle["best_value"]:
                    particle["best_value"] = fitness
                    particle["best_position"] = particle["position"].copy()

                if fitness < global_best_value:
                    global_best_value = fitness
                    global_best_position = particle["position"].copy()

                # Update velocity and position
                inertia = 0.5
                cognitive = 2 * np.random.rand() * (particle["best_position"] - particle["position"])
                social = 2 * np.random.rand() * (global_best_position - particle["position"])
                particle["velocity"] = inertia * particle["velocity"] + cognitive + social
                particle["position"] += particle["velocity"]

                # Constrain within image bounds
                particle["position"] = np.clip(particle["position"], [0, 0], [self.image.width - 1, self.image.height - 1])

            # Visualize particles after each iteration
            self.clear_swarm()
            for particle in self.particles:
                x, y = particle["position"]
                dot = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="green", outline="green")
                self.swarm.append(dot)
            self.root.update()
            time.sleep(0.1)  # Pause for visualization

    def clear_swarm(self):
        for dot in self.swarm:
            self.canvas.delete(dot)
        self.swarm.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()

