import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits#
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def plot_csv_data_recognizability(csv_recognizable_path, csv_not_recognizable_path):
# Select two countries' worth of data.
    
    # Using numpy we can use the function loadtxt to load your CSV file.
    # We ignore the first line with the column names and use ',' as a delimiter.
    recognizable = np.loadtxt(csv_recognizable_path, delimiter=',', skiprows=1)
    not_recognizable = np.loadtxt(csv_not_recognizable_path, delimiter=',', skiprows=1)

    # Normalze Values
    v = recognizable[:,3]
    v2 = not_recognizable[:,3]
    recognizable[:,3] = (v - v.min()) / (v.max() - v.min())
    not_recognizable[:,3] = (v2 - v2.min()) / (v2.max() - v2.min())


    #Clustering

    df = np.append(recognizable[:,3:5],not_recognizable[:,3:5], axis=0)
    #Initialize the class object
    kmeans = KMeans(n_clusters= 4,init='k-means++', 
            max_iter=1000, n_init=1, verbose=0, random_state=1)

    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    
    u_labels = np.unique(label)
            
    #Getting the Centroids
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    
    #plotting the results:
    labels = ["too bright","problem mix","recognizable","terrain differences"]
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = labels[i],alpha = 0.6, s=10)
    print(len(centroids))
    plt.title("Visibility Clusters in Turbine Shadow Detection")
    plt.xlabel('standard deviation')
    plt.ylabel('mean')
    plt.scatter(centroids[:,0] , centroids[:,1] ,c='b', marker='^', s=20)
    plt.legend()
    plt.show()

    #normal PLot
    plt.scatter(recognizable[:,3], recognizable[:,4], s=10, c='b', marker="s", label='recognizable')
    plt.scatter(not_recognizable[:,3], not_recognizable[:,4], s=10, c='r', marker="o", label='not_recognizable')
    plt.title("Distribution of recognizability in Turbine Shadow Detection")
    plt.xlabel('standard deviation')
    plt.ylabel('mean')
    
    plt.legend(loc='upper left');   

    # Show the legend
    plt.show()


