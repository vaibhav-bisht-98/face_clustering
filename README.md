
# Description

Inspired by my oneplus gallery feature to automatically group photos of each individual present in the photos in my gallery app, I have tried to put together a similar ML pipeline. It uses a Face detector to get crop of the persons faces which are then passed to a Face recognition model to get featrure vectors for the detected faces. Finally I have used DBSCAN to cluster the features vector together and get groups of similar photos.

## Setup

Run following command to install required libraries:
```
pip install -r requirements.txt
```
## Command line arguments

|  **Parameter** | **Default Value** | **Description**|
|----------------|-------------------|----------------|
|`input_dir`| images| Path to directory with all images|
|`eps` | 0.07 | Maximum distance between samples to be considered neighbours|
|`min_samples| 1 | The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. |
|`make-dir| `False` | Flag to output images based on clusters identified|
