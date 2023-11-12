import json
import os

result_dict = {}

# Directory containing individual text files
directory = 'runs/detect/exp16/labels'

# Initialize a counter for object tracking
object_counter = 1

# Iterate through all the text files in the directory
for filename in os.listdir(directory):
    if filename.startswith("fifa") and filename.endswith(".txt"):
        with open(os.path.join(directory, filename), "r") as file:
            lines = file.readlines()
            player_coords = []

            for line in lines:
                # Assuming each line in the text file represents a detection
                # You may need to customize this part based on the format of your text files
                detection_info = line.strip().split(" ")  # Assuming format: "class x_center y_center width height"
                player_coords.append({
                    "object_id": object_counter,  # Assign a unique identifier to each object
                    "class": detection_info[0],
                    "x_center": float(detection_info[1]),
                    "y_center": float(detection_info[2]),
                    "width": float(detection_info[3]),
                    "height": float(detection_info[4])
                })

                # Increment the object tracking counter
                object_counter += 1

            # Store the player coordinates in the result_dict
            result_dict[filename] = player_coords

# Save the combined results to a JSON file
with open("combined_results.json", "w") as json_file:
    json.dump(result_dict, json_file)
