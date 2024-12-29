import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to open the image file
def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((400, 400))  # Resize image to fit the window
        img_tk = ImageTk.PhotoImage(img)
        
        # Display the image on the GUI
        panel.config(image=img_tk)
        panel.image = img_tk
        process_button.config(state=tk.NORMAL, command=lambda: kmeans_clustering(img))

# Function to apply K-means clustering on the image
def kmeans_clustering(img):
    # Convert image to numpy array and reshape it for clustering
    img_np = np.array(img)
    img_data = img_np.reshape((-1, 3))  # Reshape to 2D array (pixels, RGB)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(img_data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Assign each pixel to the nearest centroid
    clustered_img = centers[labels].reshape(img_np.shape)

    # Convert to uint8 for displaying
    clustered_img = np.uint8(clustered_img)

    # Display the clustered image
    clustered_image = Image.fromarray(clustered_img)
    clustered_image.show()

# Set up the main window
root = tk.Tk()
root.title("K-means Clustering on Image")

# Set up a label to show the image
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Add buttons to load image and apply K-means clustering
load_button = tk.Button(root, text="Load Image", command=open_image)
load_button.pack(pady=10)

process_button = tk.Button(root, text="Apply K-means Clustering", state=tk.DISABLED)
process_button.pack(pady=10)

# Run the application
root.mainloop()

