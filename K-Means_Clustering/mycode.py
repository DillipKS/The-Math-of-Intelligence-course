# Implementaion of k-Means Clustering algo for credit card fraud detection
# https://github.com/llSourcell/k_means_clustering
# Marker info - https://matplotlib.org/api/markers_api.html

import sys
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(1)
datapath = 'k_means_clustering-master/durudataset.txt'
no_of_clusters = 2

def data_process(datapath):
    # preprocess the data
    data = np.loadtxt(datapath)
    num_rows, num_features = data.shape
    
    return data, num_rows, num_features

def plot(centroids, cluster_points, test_cluster={}):
    # Plot the clustered datapoints and their centroid
    colors = ['r', 'g', 'm', 'c', 'y', 'w']
    fig, ax = plt.subplots()

    for key in cluster_points.keys():
        for point in cluster_points[key]:
            ax.plot(point[0], point[1], (colors[key] + 'o'))

    if test_cluster:
        for key in test_cluster.keys():
            for point in test_cluster[key]:
                ax.plot(point[0], point[1], (colors[key] + 's'))
        
    for centroid in centroids:
        ax.plot(centroid[0], centroid[1], 'ko')     # black dot
    plt.show()

def euclidean(v):
    # returns norm of a vector/matrix
    return np.linalg.norm(v)

def clustering(data, belongs_to, k):
    # returns k clusters of data points
    cluster_points = {}
    for i in range(k):
        cluster_points[i] = []
    for i, value in enumerate(belongs_to):
        cluster_points[value].append(data[i])

    return cluster_points

def kmeans(data, num_rows, num_features, k):
    # k-Means algorithm
    rand_no = np.random.choice(num_rows, k)
    new_centroids = [data[i] for i in rand_no]
    belongs_to = list(np.zeros(num_rows))
    error = 1 
    iterator = 0
    
    while error > 0:
        centroids = new_centroids
        for i, row in enumerate(data):
            distance = []
            for centroid in centroids:
                distance.append(euclidean(row - centroid))
            centroid_no = np.argmin(distance)
            belongs_to[i] = centroid_no
        
        # calculate no. of datapoints in each cluster
        # by calculating the frequency of items in belongs_to list
        d = {x: belongs_to.count(x) for x in belongs_to}
        cluster_counter = d.values()
        
        tmp_centroids = np.zeros((k, num_features))
        for i, row in enumerate(data):
            tmp_centroids[belongs_to[i]] += row
        tmp_centroids = [tmp_centroids[i]/float(cluster_counter[i]) for i in range(k)]

        new_centroids = tmp_centroids
        sub = np.subtract(new_centroids, centroids)
        error = euclidean(sub)
        iterator += 1
        new_centroids = [np.ndarray.tolist(i) for i in new_centroids]
    
    cluster_points = clustering(data, belongs_to, k)

    return new_centroids, cluster_points, iterator

def inference(test_data, centroids):
    # cluster new data points
    belongs_to = list(np.zeros(test_data.shape[0]))
    
    for i, data in enumerate(test_data):
        distance = []
        for centroid in centroids:
            distance.append(euclidean(data - centroid))
        belongs_to[i] = np.argmin(distance)

    return belongs_to

def main():
    data, num_rows, num_features = data_process(datapath)
    centroids, cluster_points, iterations = kmeans(data, num_rows, num_features, no_of_clusters)
    
    # Testing
    test_data = np.array([[1.8,1.8], [0.2,0.2], [0.25,1.5]])
    test_belongs = inference(test_data, centroids)
    test_cluster = clustering(test_data, test_belongs, no_of_clusters)

    # Graph plot
    print "Centroid of clusters: "
    pprint(centroids)
    print "No. of iterations: ", iterations
    plot(centroids, cluster_points, test_cluster)

if __name__ == '__main__':
    main()


'''
Result-
Centroid of clusters: 
[[1.5805824656617171, 1.5689741160857174],
 [0.2233106749885314, 0.28960446247509586]]
No. of iterations:  3
'''