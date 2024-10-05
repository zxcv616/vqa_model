import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_dominant_color(image_path, k=4):
    # load the image
    image = cv2.imread(image_path)
    
    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # reshape into 2d array (pixels)
    pixels = image.reshape(-1, 3)
    
    # use kmeans to cluster pixel colors/values
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # find the dominant color
    dominant_color = kmeans.cluster_centers_[kmeans.labels_[0]]
    
    # convert to the closest standard color (basic color name)
    closest_color = get_closest_color_name(dominant_color)
    
    return closest_color

def get_closest_color_name(rgb_color):
    # add a dictionary of common color names with RGB values (you can expand this)
    color_names = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "black": [0, 0, 0],
        "white": [255, 255, 255]
    }
    
    # calculate Euclidean distance to find closest color
    min_distance = float('inf')
    closest_color = None
    
    for color_name, rgb in color_names.items():
        distance = np.linalg.norm(np.array(rgb_color) - np.array(rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
            
    return closest_color
