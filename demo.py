
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

# random.seed("bob")

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'pink', 'silver',
          'darkgreen', 'k', 'orange', 'brown']
EPSILON = np.finfo(float).eps
MAX = np.finfo(float).max


def main():
    # do_demo()
    dataset = generate_data(70, 4, 2)
    # plot_points(dataset)
    # c, z, sse = do_kmeans(dataset, 4)
    c, z, sse = demo_kmeans(dataset, 4, plot=True, sleep=1)
    tune_k(dataset, range(1, 7))


def do_demo(param=None):
    # generates dataset with four clusters
    # param == 'g': run with good points
    # param == 'b': run with bad points
    # param == None: run with randomly selected points
    dataset = generate_4demo_data()
    plot_points(dataset)
    k = 4
    good_initial = [(5, 5), (2, 2), (8, 0), (-1, 8)]
    bad_initial = [(0, 5), (0, 4), (1, 2), (2, 1)]
    if param == 'g':
        c, z, sse = demo_kmeans(dataset, k, good_initial, plot=True)
    elif param == 'b':
        c, z, sse = demo_kmeans(dataset, k, bad_initial, plot=True)
    else:
        c, z, sse = demo_kmeans(dataset, k, plot=True)
    print sse


def do_demo2(param=None):
    dataset = generate_2demo_data()
    plot_points(dataset)
    k = 2
    initial = [(1, 5), (5, 1)]
    c, z, sse = demo_kmeans(dataset, k, initial, plot=True)
    print sse


def tune_k(dataset, k_range):
    # run kmeans on dataset for each k in k_range (k_range is a list of ints)
    # then plot the SSE for the best clustering produced by each k
    sses = []
    for k in k_range:
        print k
        c, z, sse = do_kmeans(dataset, k)
        sses.append(sse)
    plt.plot(k_range, sses)
    plt.scatter(k_range, sses)
    plt.show()


def do_kmeans(dataset, k, runs=10):
    # run kmeans on dataset 'runs' number of times (default 10)
    # choose the clustering that gives the lowest SSE
    best_sse = MAX
    for i in range(runs):
        initial = random.sample(dataset, k)
        c, z, sse = kmeans(dataset, k, initial)
        if sse < best_sse:
            best_c = c
            best_z = z
            best_sse = sse
    return best_c, best_z, best_sse


def kmeans(dataset, k, initial):
    # basic implementation of lloyd's algorithm
    change = 1
    old_sse = 0
    iteration = 1
    c = initial
    while change > EPSILON:
        z = assign_clusters(dataset, c)
        c = calculate_new_means(dataset, c, z, k)
        sse = sum_squared_errors(dataset, c, z)
        change = abs(sse - old_sse)
        old_sse = sse
        iteration += 1
    return c, z, sse


def demo_kmeans(dataset, k, initial=None, plot=False, sleep=1):
    # same as kmeans() but can pause and plot at each iteration
    if initial is None:
        # if initial points aren't specified, choose randomly from dataset
        initial = random.sample(dataset, k)
    c = initial
    change = 1
    old_sse = 0
    e = EPSILON
    if plot:
        e *= 10 ^ (4)
    i = 1
    while change > e:
        z = assign_clusters(dataset, c)
        if plot:
            if i == 1:
                block = True
            else:
                block = False
            plot_kmeans(dataset, c, z, block=block,
                        title='k-means iteration #' + str(i))
            time.sleep(sleep)
        c = calculate_new_means(dataset, c, z, k)
        if plot:
            if i == 1:
                block = True
            else:
                block = False
            plot_kmeans(dataset, c, z, block=block,
                        title='k-means iteration #' + str(i))
            time.sleep(sleep)
        sse = sum_squared_errors(dataset, c, z)
        change = abs(sse - old_sse)
        old_sse = sse
        i += 1
    plot_kmeans(dataset, c, z, block=True, title='final')
    return c, z, sse


def calculate_new_means(dataset, old_means, z, k):
    # given the dataset and cluster assignments,
    # calculate the new centers for each cluster
    c = []
    for i in range(k):
        cluster_points = [dataset[j] for j in range(len(dataset)) if z[j] == i]
        if cluster_points == []:
            # sometimes we get an empty cluster when using lloyd's algorithm.
            # even when i selected initial points from the dataset!!!
            # just ignore it and keep the old centroid
            new_mean = old_means[i]
            c.append(new_mean)
            continue
        new_mean = tuple(np.mean(np.array(cluster_points), axis=0))
        c.append(new_mean)
    return c


def sum_squared_errors(dataset, centroids, z):
    squared_errors = []
    for i in range(len(dataset)):
        point = dataset[i]
        center = centroids[z[i]]
        squared_errors.append(squared_dist(point, center))
    return sum(squared_errors)


def squared_dist(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sum((x - y) ** 2)


def assign_clusters(dataset, centroids):
    # finds closest centroid to each point in dataset,
    # returns a list of cluster assignments
    assignments = []
    # print np.array(centroids)
    for i in range(len(dataset)):
        point = dataset[i]
        squared_distances = np.sum((np.array(centroids) - point) ** 2, 1)
        assignments.append(np.argmin(squared_distances))
    return assignments


def plot_points(list_of_points):
    x = zip(*list_of_points)[0]
    y = zip(*list_of_points)[1]
    plt.scatter(x, y)
    plt.show()


def plot_kmeans(points, c, z, block=False, title=''):
    plt.clf()
    for i in range(len(points)):
        # plot data points
        point = points[i]
        cluster = z[i]
        plt.scatter(point[0], point[1], color=COLORS[cluster])
    for i in range(len(c)):
        # plot centroids
        point = c[i]
        plt.scatter(point[0], point[1], color='k', marker='o', s=70)
    plt.title(title)
    plt.draw()
    plt.show(block=block)


def plot_kmeans_3d(points, c, z, block=False, title=''):
    Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(points)):
        # plot data points
        point = points[i]
        cluster = z[i]
        ax.scatter(point[0], point[1], point[2], color=COLORS[cluster])
    for i in range(len(c)):
        # plot centroids
        point = c[i]
        ax.scatter(point[0], point[1], point[2], color='k', marker='o', s=70)
    plt.show()


def generate_4demo_data():
    np.random.seed(2)
    centers = [(-1, 5), (2, 0), (12, 8), (10, -2)]
    points = []
    for cent in centers:
        for i in range(30):
            points.append(np.random.multivariate_normal(cent, np.identity(2)))
    return points


def generate_2demo_data():
    np.random.seed(5)
    centers = [(1, 1), (5, 5)]
    points = []
    for cent in centers:
        for i in range(30):
            points.append(np.random.multivariate_normal(cent, np.identity(2)))
    return points


def generate_data(points, clusters, dimensions):
    # creates a fake dataset with clusters by picking random centers, then
    # sampling from a gaussian distribution centered on one of those centers
    centers = []
    dataset = []
    for i in range(clusters):
        centers.append(tuple(random.sample(range(100), dimensions)))
    for j in range(points):
        center = random.sample(centers, 1)[0]
        dataset.append(tuple(np.random.multivariate_normal(
                             center, np.identity(dimensions) * 10)))
    return dataset


if __name__ == "__main__":
    main()
