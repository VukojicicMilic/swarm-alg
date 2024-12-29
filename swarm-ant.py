import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor with ACO")

        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.image = None
        self.image_tk = None
        self.drawings = []  # List of drawn items
        self.paths = []  # Path pheromones for ACO

        # Buttons and inputs
        self.load_button = tk.Button(self.toolbar, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_button = tk.Button(self.toolbar, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Label(self.toolbar, text="Ant Count:").pack(side=tk.LEFT, padx=5)
        self.ant_count_entry = tk.Entry(self.toolbar, width=5)
        self.ant_count_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(self.toolbar, text="Iterations:").pack(side=tk.LEFT, padx=5)
        self.iterations_entry = tk.Entry(self.toolbar, width=5)
        self.iterations_entry.pack(side=tk.LEFT, padx=5)

        self.run_button = tk.Button(self.toolbar, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Event bindings
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)

        self.current_line = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.drawings.clear()
            self.paths.clear()

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
        self.paths.clear()

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def run_simulation(self):
        try:
            ant_count = int(self.ant_count_entry.get())
            iterations = int(self.iterations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Ant Count and Iterations must be integers.")
            return

        if not self.drawings:
            messagebox.showerror("Error", "Please draw points and lines before running the simulation.")
            return

        points = [coords for shape, coords in self.drawings if shape == "point"]
        if len(points) < 2:
            messagebox.showerror("Error", "At least two points are required for ACO.")
            return

        self.run_aco(points, ant_count, iterations)

    def run_aco(self, points, ant_count, iterations):
        num_points = len(points)

        # Distance matrix
        distances = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distances[i][j] = self.calculate_distance(points[i], points[j])

        # Initialize pheromones
        pheromones = np.ones((num_points, num_points))
        best_path = None
        best_length = float("inf")

        for iteration in range(iterations):
            all_paths = []
            path_lengths = []

            for ant in range(ant_count):
                start_point = np.random.choice(range(num_points))
                path = [start_point]

                while len(path) < num_points:
                    current = path[-1]
                    probabilities = []

                    for next_point in range(num_points):
                        if next_point not in path:
                            probabilities.append((pheromones[current][next_point] / distances[current][next_point]))
                        else:
                            probabilities.append(0)

                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()

                    next_point = np.random.choice(range(num_points), p=probabilities)
                    path.append(next_point)

                    # Visualize ant movement
                    x1, y1 = points[current]
                    x2, y2 = points[next_point]
                    self.canvas.create_line(x1, y1, x2, y2, fill="green", width=1)
                    self.root.update()
                    time.sleep(0.1)

                all_paths.append(path)

                length = sum(distances[path[i]][path[i+1]] for i in range(-1, num_points - 1))
                path_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = path

            # Update pheromones
            pheromones *= 0.9
            for path, length in zip(all_paths, path_lengths):
                for i in range(-1, num_points - 1):
                    pheromones[path[i]][path[i+1]] += 1 / length

        # Visualize best path
        for i in range(len(best_path) - 1):
            x1, y1 = points[best_path[i]]
            x2, y2 = points[best_path[i + 1]]
            self.canvas.create_line(x1, y1, x2, y2, fill="yellow", width=3)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()

