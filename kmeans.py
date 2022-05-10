"""kmeans.py"""

from random import randrange
from math import sqrt

class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""
    def __init__(self, dim):
        self.dim = dim
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()

    def recalculate(self):
        """
        Calculate the prototype as the average of the vectors of the current members,
        then store current members as previous members, and reinitialize current members.

        :return: True if cluster membership stabilized, False otherwise.
        :rtype: bool
        """

        self.prototype = [0.0 for _ in range(self.dim)] #  reinitialize prototype

        # the prototype becomes the sum of current member vectors
        for current_member in self.current_members:
            for i in range(self.dim):
                self.prototype[i] += current_member.datapoint[i]

        # the prototype is scaled by the number of current members
        if(len(self.current_members) > 0):
            for i in range(self.dim):
                self.prototype[i] /= len(self.current_members)

        # if cluster membership stabilized, return true
        if(self.previous_members == self.current_members):
            return True
        
        # if cluster membership did not stabilize, store current members as previos members,
        # and reinitialize current members, and return false
        self.previous_members = self.current_members
        self.current_members = set()
        return False

# datapoint are lists that are unhashable so they couldn't be added to sets (to current_members and previous_members) as they were
class Datapoint:
    def __init__(self, datapoint):
        self.datapoint = datapoint

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, Datapoint) and len(self.datapoint) == len(other.datapoint):
            for i in range(len(self.datapoint)):
                if(self.datapoint[i] != other.datapoint[i]):
                    return False
            return True
        return False

    def distance(self, point):
        """
        Calculate the Euclidian distance between an input vector and the vector represented by the current object.

        :param point: the input vector
        :type point: list of floats
        :return: the Euclidian distance
        :rtype: float
        """
        eucl_sum = 0
        for i in range(len(self.datapoint)):
            eucl_sum += (point[i] - self.datapoint[i])**2
        return sqrt(eucl_sum)

class KMeans:
    def __init__(self, k, traindata, testdata, dim, prefetch_threshold = 0.5):
        self.k = k                                          # number of clusters
        self.traindata = traindata 
        self.testdata = testdata
        self.dim = dim                                      # dimensionality of data points

        self.prefetch_threshold = prefetch_threshold        # Threshold above which the corresponding html is prefetched
        self.clusters = [Cluster(dim) for _ in range(k)]    # An initialized list of k clusters

        self.accuracy = 0
        self.hitrate = 0

    def find_closest_cluster(self, datapoint, min_ind = 0):
        """
        Find the cluster closest to a data point.

        :param Datapoint datapoint: the data point to be compared
        :param int min_ind: the cluster index to be taken as the initial minimum, by default equals 0
        :return: the cluster closest to the data point
        :rtype: Cluster
        """

        for i in range(self.k):
            candidate_cluster_prototype = self.clusters[i].prototype
            min_cluster_prototype = self.clusters[min_ind].prototype
            if(datapoint.distance(candidate_cluster_prototype) < datapoint.distance(min_cluster_prototype)):
                min_ind = i

        return self.clusters[min_ind]

    def random_partition(self):
        """
        Partition data points into k clusters randomly.

        :return: None
        """

        for raw_datapoint in self.traindata:
            random_cluster = self.clusters[randrange(self.k)]
            random_cluster.current_members.add(Datapoint(raw_datapoint))

    def train(self):
        """
        Train the algorithm - determine ideal prototypes defined by the average of clusters of training data points.

        :return: None
        """

        # Select an initial random partioning with k clusters
        self.random_partition()

        print("-" * 100)
        epoch = 0
        # (Re-)calculate cluster centers and (re-)generate partitions until the cluster membership stabilizes
        # Since cluster.recalculate() returns true when a clusters had stabilized, the condition will become false once all k clusters stabilized.
        while(sum([cluster.recalculate() for cluster in self.clusters]) != self.k):

            # Generate a new partition by assigning each datapoint to its closest cluster center
            overall_distance = 0
            for i in range(self.k):
                current_cluster = self.clusters[i]
                for datapoint in current_cluster.previous_members:
                    # set the initial closest cluster to be the current cluster, to reduce the probability of a loop occuring
                    closest_cluster = self.find_closest_cluster(datapoint, i)
                    overall_distance += datapoint.distance(closest_cluster.prototype)
                    closest_cluster.current_members.add(datapoint)
            
            epoch += 1
            print("| epoch", epoch, "| total distance:", overall_distance, "| # in each cluster:", [len(cluster.current_members) for cluster in self.clusters], "|")
            print("-" * 100)

    def get_cluster_of(self, index):
        """
        Get the estimated cluster of a client.

        :param int index: the index referring to the client's data in the train and test sets
        :return: the estimated cluster
        :rtype: Cluster
        """

        requested_datapoint = Datapoint(self.traindata[index])
        for cluster in self.clusters:
            for datapoint in cluster.current_members:
                if(datapoint == requested_datapoint):
                    return cluster

    def test(self):
        """
        Calculate hitrate and accuracy.

        :return: None
        """

        n_prefectched_htmls = 0     # total number of prefetched htmls for all clients
        requests = 0                # total number of requests made by all clients
        hits = 0                    # total number of hits for all clients

        # iterate along all clients
        for i in range(len(self.testdata)):

            # calculate the prediction by making the values above the prefetch threshold 1, and below 0
            estimated_prototype = self.get_cluster_of(i).prototype
            prediction = [(0 if i < self.prefetch_threshold else 1) for i in estimated_prototype]

            # add the number of htmls prefetched by this prediction to the total number of prefetched htmls
            n_prefectched_htmls += sum(prediction)

            # count requests and hits
            for j in range(self.dim):
                if(self.testdata[i][j] == 1):
                    requests  += 1
                    if(prediction[j] == 1):
                        hits += 1


        # calculate hit rate and accuracy
        if(requests > 0):
            self.hitrate = hits / requests
        if(n_prefectched_htmls > 0):
            self.accuracy = hits / n_prefectched_htmls

        print(n_prefectched_htmls, "prefetched htmls,", requests, "requests made by the client,", hits, "hits (client requests correctly guessed)")

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i, cluster in enumerate(self.clusters):
            print("Members cluster["+str(i)+"] :", cluster.current_members)
            print()

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster["+str(i)+"] :", cluster.prototype)
            print()