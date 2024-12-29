import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import convolve, maximum_filter
from sklearn.cluster import DBSCAN

class TerrainAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Terrain Analyzer with Convolution and Region Detection")

        # GUI elements
        self.canvas = None
        self.image = None
        self.image_path = None

        load_btn = tk.Button(root, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=10, pady=10)

        analyze_btn = tk.Button(root, text="Analyze", command=self.analyze_image)
        analyze_btn.pack(side=tk.LEFT, padx=10, pady=10)

        reset_btn = tk.Button(root, text="Reset", command=self.reset)
        reset_btn.pack(side=tk.LEFT, padx=10, pady=10)

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

        # Convert the image to grayscale for simplicity
        height_map = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Apply a simple convolution (Gaussian filter for smoothing or Sobel for edges)
        kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])  # Simple edge detection filter
        convolved_image = convolve(height_map, kernel)

        # Find local maxima using maximum_filter
        local_maxima = self.find_local_maxima(convolved_image)

        # Find regions around the maxima
        regions = self.find_maxima_regions(local_maxima)

        # Display results
        self.display_results(regions)

    def find_local_maxima(self, image):
        # Use a maximum filter to find local maxima
        neighborhood_size = 20  # Size of the neighborhood to check for maxima
        local_maxima = maximum_filter(image, size=neighborhood_size) == image
        return np.argwhere(local_maxima)

    def find_maxima_regions(self, local_maxima):
        # DBSCAN clustering to group nearby maxima into regions
        if len(local_maxima) == 0:
            return []

        # Use DBSCAN to cluster nearby maxima into regions
        clustering = DBSCAN(eps=20, min_samples=1).fit(local_maxima)  # Adjust eps for your image scale
        labels = clustering.labels_

        # Group regions based on clusters
        regions = []
        for cluster_id in np.unique(labels):
            # Get coordinates of maxima in this cluster
            cluster_coords = local_maxima[labels == cluster_id]
            if len(cluster_coords) > 1:
                # Calculate the bounding box for the cluster
                x_min = np.min(cluster_coords[:, 1]) - 10
                x_max = np.max(cluster_coords[:, 1]) + 10
                y_min = np.min(cluster_coords[:, 0]) - 10
                y_max = np.max(cluster_coords[:, 0]) + 10
                regions.append(((x_min, y_min), (x_max, y_max)))

        return regions

    def display_results(self, regions):
        if len(regions) == 0:
            messagebox.showinfo("Result", "No regions found!")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.image)

        # Plot regions as rectangles
        for (x_min, y_min), (x_max, y_max) in regions:
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                       fill=False, edgecolor='blue', linewidth=2))

        ax.axis('off')

        # Clear existing canvas and display results without a legend
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

