"""run_clustering.py -- the main program that does the data IO (by reading 
the 4 data files),handles the cmd line interaction with the programmer 
and calls three clustering methods."""

import sys
from kmeans import KMeans
from kohonen import Kohonen

import matplotlib.pyplot as plt

def main():
    if len(sys.argv) == 5:
        train, test, requests, clients, dim = read_data(*sys.argv[1:])
    else:
        print("No files where defined (python run_clustering.py [traindata, testdata, requests, clients]), using defaults")
        train, test, requests, clients, dim = read_data()

    clustering_algorithm = None
    while True:
        try:
            algorithm = int(input("Run K-means (1), Kohonen SOM (2), Quit (3), or Test parameters for K-means (4), or Test parameter for Kohonen (5) ? "))
        except ValueError:
            continue
            
        if algorithm == 1:
            clustering_algorithm = kmeans_init(train, test, dim)
        elif algorithm == 2:
            clustering_algorithm = kohonen_init(train, test, dim)
        elif algorithm == 3:
            exit()
        elif algorithm == 4:
            kmeans_param_test(train, test, dim)
            break
        elif algorithm == 5:
            kohonen_param_test(train, test, dim)
            break
        else:
            continue

        if not clustering_algorithm:
            continue

        input("Perform the actual training! (hit enter)")
        print("Training ...")
        clustering_algorithm.train()
        print("Training finished!")
        input("Perform the testing! (hit enter)")
        print("Testing ...")
        clustering_algorithm.test()
        print("Testing finished!")

        while True:
            try:
                output = int(input("Show output print_test(1), vector members(2), vector prototypes(3), Quit(4) or set prefetch threshold(5)? "))
            except ValueError:
                continue

            if output == 1:
                clustering_algorithm.print_test()
            elif output == 2:
                clustering_algorithm.print_members()
            elif output == 3:
                clustering_algorithm.print_prototypes()
            elif output == 4:
                break
            elif output == 5:
                try:
                    prefetch_threshold = float(input("Prefetch threshold = "))
                    clustering_algorithm.prefetch_threshold = prefetch_threshold
                    print("Testing algorithm with newly set prefetch threshold...")
                    clustering_algorithm.test()
                    print("Testing finished!")
                except Exception as e:
                    print("ERROR while setting prefetch threshold: ", e)
                    exit()

class SettingTest:

    def __init__(self, title):
        """
        Constructor.

        :param string title: "Hit rate" / "Accuracy" / "Hit rate + Accuracy"
        :return: None
        """
        self.title = title
        self.max_value = 0                                              # maximum value of the setting
        self.max_k = 0                                                  # num of clusters at the maximum value
        self.max_prefetch_threshold = 0                                 # prefetch threshold at the maximum value
        self.values = [[0 for pre in range(0, 9)] for k in range(0, 19)]     # all values (2D array)

    def add_value(self, value, k_ind, pre_ind):
        """
        Add a new value (hitrate, accuracy, or their combination) for the given parameter settings.
        
        :param int value: hitrate/accuracy/their sum
        :param int k_ind: cluster index = the number of clusters -2
        :param int pre_ind: prefetch threshold index = prefetch threshold * 10 - 1
        :return: None
        """
        self.values[k_ind][pre_ind] = value                             # store the value
        if(value > self.max_value):                                     # if the value if greater than the current max,
            self.max_value = value                                      # then change the max and the corresponding parameter settings
            self.max_k = k_ind + 2
            self.max_prefetch_threshold = (pre_ind + 1) / 10

    def display(self):
        """
        Display a heatmap of the collected values on each combination of the parameter settings (prefetch threshold and num of clusters), then print the maximum and its parameter settings.

        :return: None
        """
        plt.imshow(self.values, cmap='viridis')
        plt.colorbar()
        plt.xlabel("Prefetch threshold")
        plt.ylabel("Number of clusters")
        plt.title(self.title)
        plt.show()

        message = '{} max = {}, at\n\tprefetch threshold = {}\n\tnumber of clusters = {}\n'
        print(message.format(self.title, self.max_value, self.max_prefetch_threshold, self.max_k))
        

def kmeans_param_test(train, test, dim):
    """
    Train the K-means algorithm on each combination of prefetch threshold (range: 0.1, 0.9, in increments of 0.1) and number of clusters (range: 2, 20, in increments of 1),
    and test it on hit rate, accuracy, and their combination (sum).
    Output: Heat maps and maximum hit rate, accuracy, and combination with their respective parameter settings.

    :param float[][] train: train data
    :param float[][] test: test data
    :param int dim: data dimensionality
    :return: None
    """
    hitrate = SettingTest("Hit rate")
    accuracy = SettingTest("Accuracy")
    sum_hitrate_accuracy = SettingTest("Hit rate + Accuracy")

    for k in range(2, 21):
        for pre in range(1, 10):
            prefetch_threshold = pre / 10
            clustering_algorithm = KMeans(k, train, test, dim, prefetch_threshold)
            clustering_algorithm.train()
            clustering_algorithm.test()
            hitrate.add_value(clustering_algorithm.hitrate, k-2, pre-1)
            accuracy.add_value(clustering_algorithm.accuracy, k-2, pre-1)
            sum_hitrate_accuracy.add_value(clustering_algorithm.accuracy + clustering_algorithm.hitrate, k-2, pre-1)

    hitrate.display()
    accuracy.display()
    sum_hitrate_accuracy.display()

def kohonen_param_test(train, test, dim):
    """
    Train the K-means algorithm on each combination of prefetch threshold (range: 0.1, 0.9, in increments of 0.1) and number of clusters (range: 2, 20, in increments of 1),
    and test it on hit rate, accuracy, and their combination (sum).
    Output: Heat maps and maximum hit rate, accuracy, and combination with their respective parameter settings.

    :param float[][] train: train data
    :param float[][] test: test data
    :param int dim: data dimensionality
    :return: None
    """
    hitrate = SettingTest("Hit rate")
    accuracy = SettingTest("Accuracy")
    sum_hitrate_accuracy = SettingTest("Hit rate + Accuracy")

    for k in range(2, 21):
        for pre in range(1, 10):
            prefetch_threshold = pre / 10
            clustering_algorithm = Kohonen(k, epochs, train, test, dim)
            clustering_algorithm.train()
            clustering_algorithm.test()
            hitrate.add_value(clustering_algorithm.hitrate, k-2, pre-1)
            accuracy.add_value(clustering_algorithm.accuracy, k-2, pre-1)
            sum_hitrate_accuracy.add_value(clustering_algorithm.accuracy + clustering_algorithm.hitrate, k-2, pre-1)

    hitrate.display()
    accuracy.display()
    sum_hitrate_accuracy.display()

def read_data(train_filename="train.dat", test_filename="test.dat",
              requests_filename="requests.dat", clients_filename="clients.dat"):
    train, dim = read_train(train_filename)
    test = read_test(test_filename, dim)
    requests = read_requests(requests_filename)
    clients = read_clients(clients_filename)

    if dim != len(requests):
        print("ERROR: the number of dimensions in the training data does not match the number of requests in " + requests_filename)
        exit()

    return train, test, requests, clients, dim

def read_train(filename):
    train_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                train_data.append(list(map(float, line.rstrip("\n").split())))
    except Exception as e:
        print("Error while reading train data: ", e)
        exit()
        
    dim = 0
    for data in train_data:
        if dim == 0:
            dim = len(data)
        else:
            if dim != len(data):
                print("ERROR: Varying dimensions in train data.")
                exit()

    return train_data, dim

def read_test(filename, dim):
    test_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                test_data.append(list(map(float, line.rstrip("\n").split())))
    except Exception as e:
        print("Error while reading test data: ", e)
        exit()

    for data in test_data:
        if len(data) != dim:
            print("ERROR: Dimensions in test data do not correspond to those in the train data.")
            exit()

    return test_data

def read_requests(filename):
    request_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                request_data.append(line.rstrip("\n"))
    except Exception as e:
        print("Error while reading requests data: ", e)
        exit()
        
    return request_data

def read_clients(filename):
    clients = []
    try:
        with open(filename, "r") as f:
            for line in f:
                clients.append(line.rstrip("\n"))
    except Exception as e:
        print("Error while reading clients data: ", e)
        exit()
        
    return clients

def kmeans_init(train, test, dim):
    k = None
    while not isinstance(k, int):
        try:
            k = int(input("How many clusters (k) ? "))
            return KMeans(k, train, test, dim)
        except Exception as e:
            print("ERROR while trying to initialize KMeans: ", e)

def kohonen_init(train, test, dim):
    n = None
    while not isinstance(n, int):
        try:
            n = int(input("Map size (N*N) ? "))
        except Exception as e:
            print("ERROR while trying to initialize Kohonen: ", e)

    epochs = None
    while not isinstance(epochs, int):
        try:
            epochs = int(input("Number of training epochs ? "))
            return Kohonen(n, epochs, train, test, dim)
        except Exception as e:
            print("ERROR while trying to initialize Kohonen: ", e)


if __name__ == "__main__":
    main()