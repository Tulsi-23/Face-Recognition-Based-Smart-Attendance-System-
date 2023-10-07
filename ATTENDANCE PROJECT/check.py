import pickle
import numpy as np
import cv2
# Load data from 'faces_data.pkl'
with open('data/faces_data.pkl', 'rb') as faces_file:
    faces_data = pickle.load(faces_file)
new_height = 50
new_width = 50

# Reshape each image to (new_height, new_width)
reshaped_faces_data = []
for image in faces_data:
    resized_image = cv2.resize(image, (new_width, new_height))
    reshaped_faces_data.append(resized_image)

# Convert the list of reshaped images to a numpy array
reshaped_faces_data = np.array(reshaped_faces_data)

# Now, reshaped_faces_data will have the shape (n_samples, new_height, new_width, n_channels)
# Save the reshaped data to a new pickle file
with open('data/reshaped_faces_data.pkl', 'wb') as reshaped_faces_file:
    pickle.dump(reshaped_faces_data, reshaped_faces_file)
