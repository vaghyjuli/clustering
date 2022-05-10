import random
import math


class Cluster:
    """This class represents the clusters, it contains the
    prototype and a set with the ID's (which are Integer objects) 
    of the datapoints that are member of that cluster."""

    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()


class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(dim) for _ in range(n)] for _ in range(n)]

        # Step 1: initialize map with random vectors
        # randomly initialize cluster center with weight values between 0.0 and 1.0
        self.prototype = [random.random() for _ in range(dim)]

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8

        self.initial_square_size = len(self.clusters)
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def find_bmu(self, input_vector):

        row = 0
        column = 0
        bmu = self.clusters[0][0].prototype

        # loop through the map
        for i in range(self.n):
            for j in range(self.n):
                # if the euclidean distance between the input vector and the current node is less than the current BMU
                # then assign it as the new BMU and obtain its coordinates
                if math.dist(self.clusters[i][j].prototype, input_vector) < math.dist(bmu, input_vector):
                    row = i
                    column = j
                    bmu = self.clusters[i][j].prototype

        return row, column

    def update_weights(self, input_vector, row, column, bmu_neighborhood, learning_rate):
        # go through each node in the bmu's neighborhood
        for i in range(row - int(bmu_neighborhood), row + int(bmu_neighborhood), 1):
            for j in range(column - int(bmu_neighborhood), column + int(bmu_neighborhood), 1):
                if 0 <= i < self.n and 0 <= j < self.n:

                    # use the formula of the update weights: (1 - learning_rate) * the current weight in the grid
                    self.clusters[i][j].prototype = [(1 - learning_rate) * element for element in self.clusters[i][j].prototype]

                    # sum the learning rate and value of the input vector
                    sum = []
                    for element in input_vector:
                        sum.append(learning_rate * input_vector[int(element)])

                    # go through each element of the grid and update the weight
                    for index in range(len(self.clusters[i][j].prototype)):
                        self.clusters[i][j].prototype[i] += sum[index]

    def train(self):

        # Step 2: Calculate the square size and the learning rate, these decrease linearly with the number of epochs.

        start_learning_rate = self.initial_learning_rate
        start_square_size = self.initial_square_size
        epochs = self.epochs

        learning_rate = start_learning_rate
        bmu_neighborhood = start_square_size

        # iterate through each epoch
        for epoch in range(epochs):

            #  Step 3: Every input vector is presented to the map (always in the same order)
            # iterate through each input_vector in the training data
            for training_data in range(len(self.traindata)):

                input_vector = self.traindata[training_data]

                #  For each vector its Best Matching Unit is found, and :
                #  Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.

                # find BMU of given input vector
                row, column = self.find_bmu(input_vector)

                # update weights of the nodes in the BMU's neighbourhood
                self.update_weights(input_vector, row, column, bmu_neighborhood, learning_rate)

            # Step 2: Calculate the square size and the learning rate, these decrease linearly with the number of epochs.
            learning_rate = start_learning_rate * (1 - (epoch / epochs))
            bmu_neighborhood = (start_square_size / 2) * (1 - (epoch / epochs))

        # Since training kohonen maps can take quite a while, presenting the user with a progress bar would be nice


    def test(self):

        n_prefectched_htmls = 0  # total number of prefetched htmls for all clients
        requests = 0  # total number of requests made by all clients
        hits = 0  # total number of hits for all clients
        estimated_prototype = self.clusters[0][0].prototype

        # iterate along all clients
        for i in range(len(self.testdata)):
            # for each client find the cluster of which it is a member
            for x in range(self.n):
                for y in range(self.n):
                    if self.clusters[x][y].prototype == self.testdata[i]:
                        estimated_prototype = self.clusters[x][y].prototype

            prediction = [(0 if i < self.prefetch_threshold else 1) for i in estimated_prototype]

            # add the number of htmls prefetched by this prediction to the total number of prefetched htmls
            n_prefectched_htmls += sum(prediction)

            # count requests and hits
            for j in range(self.n):
                if self.testdata[i][j] == 1:
                    requests += 1
                    if prediction[j] == 1:
                        hits += 1

        # calculate hit rate and accuracy
        if requests > 0:
            self.hitrate = hits / requests
        if n_prefectched_htmls > 0:
            self.accuracy = hits / n_prefectched_htmls

        print(n_prefectched_htmls, "prefetched htmls,", requests, "requests made by the client,", hits,
              "hits (client requests correctly guessed)")


    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate + self.accuracy)
        print()

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster[" + str(i) + "][" + str(j) + "] :", self.clusters[i][j].current_members)
                print()

    def print_prototypes(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Prototype cluster[" + str(i) + "][" + str(j) + "] :", self.clusters[i][j].prototype)
                print()
