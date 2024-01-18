#Nikhil Das Karavatt
#1002085391

import pandas as pd #importing pandas which for easy data read and manipulation
import numpy as np #importing numpy for mathematical calulations
import matplotlib.pyplot as plt #importing matplotlib for display of the graph
import argparse #importing argparse to add arguments for the .py file
import random #importing random for initialization of centroids

def read_file(path): #made function to read the file
    df = pd.read_csv(path,delimiter='\s+', header=None) #reading the text file 
    features = df.iloc[:,:-1] #dropping the target and taking only features
    return features #return features

def initial_centroids(features, no_of_clusters): #made function to initial centroids
    random.seed(0) #initial seed to 0 to have a constant value always
    indices = random.sample(range(len(features)), no_of_clusters) #finding indices using randomness
    centroids = features.iloc[indices].values.tolist() #assign particular features as centroids
    centroids = pd.DataFrame(centroids).T #converting to dataframe for further use
    return centroids #returning centroids which was initialzed

def get_cluster_name(features,centroids): #function to get assign cluster to different points
    euclidean_distance = centroids.apply(lambda a: np.sqrt(((features - a)**2).sum(axis=1))) #finding euclidean distance between points and centroids 
    cluster_name = euclidean_distance.idxmin(axis=1) #assigning the point to the cluster from which it has the least distance
    return cluster_name #returning the cluster name

def updating_centroids(features,cluster_name): #function to update the centroids once the centroids have been initialized
    centroids = features.groupby(cluster_name).apply(lambda a : a.mean()).T #finding the arthemetic mean distance between points and centroids and updating it to cluster with smaller values
    return centroids #returning the updated centroids

def loss(features,centroids,cluster_name): #made the loss function to find the error
    centroids =centroids.T #Transpose the cetroids for calculations
    total_error = 0 #initializing the total error with 0
    for i,centroid in enumerate(centroids.itertuples(index=False)): #for loop through index and values of centroids without getting its index value back
        cluster_points = features[cluster_name==i] #taking features which belongs to the same cluster
        error = np.sqrt(((cluster_points - centroid)**2).sum(axis=1)) #finding the euclidean distance of a data point from their centroid
        total_error += error.sum() # adding the euclidean distance of all data points to give one value of error
    return total_error #returning the total error

def plot(error_vs_no_of_clusters): # fucntion to plot the graph
    x = np.array(range(2, 11)) #assign the value of x  which is the different k values
    y = np.array(error_vs_no_of_clusters) #assign the values of y which is different error values at differe k times
    plt.style.use("dark_background") #giving the black background to graph
    plt.figure(figsize=(12, 6)) #fixing the size of the plot 
    plt.plot(x, y, marker='o',color='#00FFFF',linewidth=2, alpha = 1) #plotting the line graph which represents k vs error
    plt.plot(x , y, label='Shadow', color='#00FFFF', linewidth=8, alpha=0.2) #plotting a line to act like a show for better visualization
    for i, (xi, yi) in enumerate(zip(x, y)): #loop to go through each values of k and error
        plt.text(xi, yi, f'({xi}, {round(yi,2)})', color='#00FFFF', fontsize=8, ha='right', va='top') #printing the value on graph
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='gray') #providing grid to the graph
    plt.title('Error vs Number of Clusters (k)',color='white') #providing the title of the graph
    plt.xlabel('Number of Clusters (k)',color='#FFFF00') #name of the x axis
    plt.ylabel('Error',color='#FFFF00') #name of the y axis
    return plt.show() #returning to show the graph

def main(path): #defining the main function
    features = read_file(path) #reading the file
    error_vs_no_of_clusters = [] #creating the list to store to store error at different k values
    for i in range(2,11): #running the k through different values from 2 to 10
        max_iterations = 20 #defining the max iteration to be 20
        no_of_clusters =i #assigning the k value
        centroids = initial_centroids(features,no_of_clusters) #initailizing the centroid
        previous_centroids = pd.DataFrame() #creating a another centroid variable to be used to be compare with its previous one
        iteration = 1 #initalizing iteration to be 1
        while iteration <= max_iterations and not centroids.equals(previous_centroids): #creating a while loop with two condition one matching the iteration and other checking centroid with previous one
            previous_centroids = centroids #assignment it to previous one
            cluster_name = get_cluster_name(features,centroids) #getting the labels for the cluster
            centroids = updating_centroids(features,cluster_name) #updating the centroid to the new ones
            iteration += 1 #incrementing the iteraion by one
        error= loss(features,centroids,cluster_name) #finding the error for that k value
        error_vs_no_of_clusters.append(error)#append that k value error to the error list
        print(f"For k = {no_of_clusters} the error is {error:.4f}") #printing the error for k value after completion of iteration
    plot(error_vs_no_of_clusters) #plotting graph  of error vs number of cluster k

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="K-Means Clustering Script to process clustering of data from a file.") #calling modue to add argument
    parser.add_argument("Data_File", help="Path to the data file") #adding argument
    argument = parser.parse_args() #parsing the argument
    main(argument.Data_File) #calling the main function 