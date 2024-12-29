import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor with Grey Wolf Optimizer")

        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.image = None
        self.image_tk = None
        self.drawings = []  # List of drawn items (points and lines)
        self.swarm = []  # Grey Wolf visualizations

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

        self.run_button = tk.Button(self.toolbar, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Event bindings for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)

        self.wolves = []
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

    def calculate_fitness(self, position):
        fitness = 0
        # Iterate over the shapes (points and lines) in the drawings list
        for shape, coords in self.drawings:
            if shape == "point":
                px, py = coords
                distance = np.linalg.norm(np.array(position) - np.array([px, py]))
                fitness += distance
            elif shape == "line":
                # For each line segment, calculate the fitness based on the nearest point on the line
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    line_vec = np.array([x2 - x1, y2 - y1])
                    point_vec = np.array(position) - np.array([x1, y1])
                    line_len = np.linalg.norm(line_vec)
                    if line_len > 0:
                        line_unit_vec = line_vec / line_len
                        proj_length = np.dot(point_vec, line_unit_vec)
                        proj_length = max(0, min(line_len, proj_length))
                        closest_point = np.array([x1, y1]) + proj_length * line_unit_vec
                        distance = np.linalg.norm(position - closest_point)
                        fitness += distance / (len(coords) - 1)
        return fitness

    def run_simulation(self):
        try:
            swarm_size = int(self.swarm_size_entry.get())
            iterations = int(self.iterations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Swarm Size and Iterations must be valid numbers.")
            return

        if not self.image:
            messagebox.showerror("Error", "Please load an image before running the simulation.")
            return

        self.run_gwo(swarm_size, iterations)

    def run_gwo(self, swarm_size, iterations):
        # Initialize Wolves with random positions
        self.wolves = [
            {
                "position": np.array([
                    np.random.randint(0, self.image.width),
                    np.random.randint(0, self.image.height)
                ], dtype=np.float64),
                "fitness": float("inf")
            }
            for _ in range(swarm_size)
        ]

        for iteration in range(iterations):
            # Sort wolves by fitness (best wolf has lowest fitness)
            for wolf in self.wolves:
                wolf["fitness"] = self.calculate_fitness(wolf["position"])
            self.wolves.sort(key=lambda wolf: wolf["fitness"])

            # Get the positions of the best wolves (alpha, beta, delta)
            alpha_wolf = self.wolves[0]
            beta_wolf = self.wolves[1]
            delta_wolf = self.wolves[2]

            # Update positions of all wolves based on GWO equations
            for wolf in self.wolves:
                a = 2 - iteration * (2 / iterations)  # Linearly decreasing parameter

                # Update position using alpha, beta, and delta wolves
                for i in range(2):  # 2D space (x, y)
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * r1 - a
                    C1, C2, C3 = 2 * r1, 2 * r2, 2 * r1
                    D_alpha = np.abs(C1 * alpha_wolf["position"][i] - wolf["position"][i])
                    D_beta = np.abs(C2 * beta_wolf["position"][i] - wolf["position"][i])
                    D_delta = np.abs(C3 * delta_wolf["position"][i] - wolf["position"][i])

                    wolf["position"][i] = alpha_wolf["position"][i] - A1 * D_alpha
                    wolf["position"][i] = beta_wolf["position"][i] - A2 * D_beta
                    wolf["position"][i] = delta_wolf["position"][i] - A3 * D_delta

                # Clip wolf positions within the image bounds
                wolf["position"] = np.clip(wolf["position"], [0, 0], [self.image.width - 1, self.image.height - 1])

            # Visualization of Wolves
            self.clear_swarm()
            for wolf in self.wolves:
                x, y = wolf["position"]
                dot = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="green", outline="green")
                self.swarm.append(dot)

            # Visualization of Drawn Points and Lines
            for shape, coords in self.drawings:
                if shape == "point":
                    px, py = coords
                    self.canvas.create_oval(px-2, py-2, px+2, py+2, fill="red", outline="red")
                elif shape == "line":
                    for i in range(len(coords) - 1):
                        x1, y1 = coords[i]
                        x2, y2 = coords[i + 1]
                        self.canvas.create_line(x1, y1, x2, y2, fill="red")

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

