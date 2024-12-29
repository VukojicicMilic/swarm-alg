import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time


class GaussianProcess:
    def __init__(self, kernel, noise=1e-5):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.Y_train = None

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_s = self.kernel(self.X_train, X)
        K_ss = self.kernel(X, X) + 1e-8 * np.eye(len(X))  
        K_inv = np.linalg.inv(K)
        mu = K_s.T.dot(K_inv).dot(self.Y_train)
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu, np.diag(cov)


class RBFKernel:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * sqdist / self.length_scale**2)


class DragonflyOptimizer:
    def __init__(self, objective_function, bounds, kernel, acquisition_function="UCB", noise=1e-5):
        self.objective_function = objective_function
        self.bounds = bounds
        self.kernel = kernel
        self.acquisition_function = acquisition_function
        self.gp = GaussianProcess(kernel)
        self.X_sample = []
        self.Y_sample = []

    def acquisition(self, X, gp):
        mu, sigma = gp.predict(X)
        kappa = 2.0
        return mu + kappa * sigma

    def optimize(self, num_iterations, swarm_size, random_initial_points=5):
        swarm_positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(swarm_size, self.bounds.shape[0]))

        for iteration in range(num_iterations):
            # Acquisition step: find best acquisition value
            acquisition_values = np.array([self.acquisition(x.reshape(1, -1), self.gp) for x in swarm_positions])
            best_index = np.argmax(acquisition_values)
            best_position = swarm_positions[best_index]
            best_value = self.objective_function(best_position)

            # Visualize the swarm particles
            self.update_swarm_positions(swarm_positions)

            # Update particles towards optimal solution
            swarm_positions += np.random.randn(swarm_size, self.bounds.shape[0]) * 0.1  # random movement
            swarm_positions = np.clip(swarm_positions, self.bounds[:, 0], self.bounds[:, 1])

            # Update GP with new data
            self.X_sample = np.vstack([self.X_sample, best_position])
            self.Y_sample = np.append(self.Y_sample, best_value)
            self.gp.fit(self.X_sample, self.Y_sample)

            print(f"Iteration: {iteration + 1}, Best value: {best_value} at {best_position}")
            time.sleep(0.1)  # Pause for visualization
        return best_position, best_value

    def update_swarm_positions(self, swarm_positions):
        self.clear_swarm()
        for pos in swarm_positions:
            x, y = pos
            dot = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="green", outline="green")
            self.swarm.append(dot)
        self.root.update()

    def clear_swarm(self):
        for dot in self.swarm:
            self.canvas.delete(dot)
        self.swarm.clear()


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor with DA")

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

        self.algorithm_selector = ttk.Combobox(self.toolbar, values=["DA"], state="readonly")
        self.algorithm_selector.set("DA")
        self.algorithm_selector.pack(side=tk.LEFT, padx=5)

        self.run_button = tk.Button(self.toolbar, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Event bindings
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)

        # Settings for Dragonfly Algorithm
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

        if algorithm == "DA":
            self.run_dragonfly(swarm_size, iterations)
        else:
            messagebox.showerror("Error", "Algorithm not implemented.")

    def run_dragonfly(self, swarm_size, iterations):
        if not self.image:
            messagebox.showerror("Error", "Please load an image before running the simulation.")
            return

        def objective_function(x):
            return self.calculate_fitness(x)

        bounds = np.array([[0, self.image.width - 1], [0, self.image.height - 1]])
        kernel = RBFKernel(length_scale=1.0)
        optimizer = DragonflyOptimizer(objective_function, bounds, kernel, acquisition_function="UCB", noise=1e-5)
        best_point, best_value = optimizer.optimize(num_iterations=iterations, swarm_size=swarm_size)

        print(f"Best found point: {best_point}")
        print(f"Best found value: {best_value}")

        # Visualize the best point
        x, y = best_point
        dot = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="green", outline="green")
        self.swarm.append(dot)
        self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()

