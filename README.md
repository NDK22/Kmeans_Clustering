# K-Means Clustering on UCI Datasets

## Overview

This repository contains the implementation of K-means clustering. The goal is to implement the K-means clustering algorithm on UCI datasets without using any library functions.

## Task Description

In this task, the objective is to implement the K-means clustering algorithm with the following specifications:

- **Dataset Description:** The data file follows the same format as the training files in the UCI datasets directory. The files contain features, and the last column contains class labels. The task involves clustering based on features, excluding class labels.

- **K-means Algorithm:**
  1. Initialize 'K', the number of clusters to be created.
  2. Randomly assign K centroid points.
  3. Assign each data point to its nearest centroid to create K clusters.
  4. Re-calculate the centroids using the newly created clusters.
  5. Repeat steps 3 and 4 until the centroid gets fixed.

- **Initialization:** Implement different initialization approaches to address the Initial Centroid Problem. Avoid using library functions for initialization. Use random state 0.

- **Tasks:** The program takes one argument `<data_file>`, the path name of a file. Given a dataset (yeast/pendigits/satellite), it initializes K-means clustering, runs clustering for a range of K values (2-10), prints the error for each K value after 20 iterations, and displays a graph of Error vs K.

- **Error Calculation:**
  The Error is calculated as follows:
Error = Σ (Sum of Sqaured Error (xn, μk))

where xn is a data point and μk is the centroid of its assigned cluster.

python kmeans.py <data_file>
Replace <data_file> with the path to your dataset file.

License
This project is licensed under the MIT License - see the LICENSE file for details.
