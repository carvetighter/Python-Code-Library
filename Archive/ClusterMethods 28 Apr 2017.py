###############################################################################################
###############################################################################################
#
# This import file <ClusterMethods.py> contains methods to cluster data
#
# Requirements:
# package numpy
# package warnings
# package time
# package sklearn.cluster
#
# Methods included:
# AgglClust()
#    - Agglomerative clustering algorithm
#
# BatchKMClust()
#    - K-Nearest Neighbors batch clustering algorithm
#
# DbsClust()
#    - Density clustering aglorithm
#
# SpectClust()
#    - Spectral clustering algorithm
#
# FixDbscanRsults()
#    - corrects for the '-1' values that may occur from the Density clustering algorithm
#
# SwitchCluster()
#    - switcher function for the clustering algorithms
#
# CreateClusterPrediction()
#    - creates the object to cluster the data using the switcher function
#
# GetClusterResults()
#    -  fit and predicts the clusters based on the algorithms passed
#
# Important Info:
# None
###############################################################################################
###############################################################################################

# package import
import numpy, warnings, time
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

# ignore all warnings
warnings.simplefilter(action = 'ignore')

def AgglClust(list_c_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns an Agglomerative Clustering Object based on the list of arguements passed; this method is 
    # designed to find the most appropriate type of clustering method for the data
    #
    # Requirements:
    # package sklearn.cluster.AgglomeativeClustering
    #
    # Inputs:
    # list_c_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # list_c_args[0] -> number of clusters, <int>
    # list_c_args[1] -> affinity, <string>
    # list_c_args[2] -> linkage, <string>
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Agglomerative Clustering Object
    # Desc: this is an object which clusters the group of data bassed to the object
    ###############################################################################################
    ###############################################################################################

    return AgglomerativeClustering(n_clusters = list_c_args[0], affinity = list_c_args[1], linkage = list_c_args[2])

def BatchKmClust(list_c_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns an Batch KMeans Clustering object (KNN) based on the list of arguements passed; this method is 
    # designed to find the most appropriate type of clustering method for the data
    #
    # Requirements:
    # package sklearn.cluster.MiniBatchKMeans
    #
    # Inputs:
    # list_c_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # list_c_args[0] -> number of clusters, <int>
    # list_c_args[1] -> initialize cluster, <string>
    # list_c_args[2] -> number of initial passes to test the initial cluster, <int>
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Batch KMeans Clustering Object
    # Desc: this is an object which clusters the group of data bassed to the object
    ###############################################################################################
    ###############################################################################################

    return MiniBatchKMeans(n_clusters = list_c_args[0], init = list_c_args[1], n_init = list_c_args[2], batch_size=1000, init_size = 3000)

def DbsClust(list_c_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns an Densitiy Based Scan (DBSCAN) clustering object based on the list of arguements passed; 
    # this method is designed to find the most appropriate type of clustering method for the data
    #
    # Requirements:
    # package sklearn.cluster.DBSCAN
    #
    # Inputs:
    # list_c_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # list_c_args[0] -> number of minimum samples to be a core point, <int>
    # list_c_args[1] -> eps, <float>
    # list_c_args[2] -> distance metric, <string>
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: DBSCAN object
    # Desc: this is an object which clusters the group of data bassed to the object
    ###############################################################################################
    ###############################################################################################

    return DBSCAN(min_samples = list_c_args[0], eps = list_c_args[1], metric = list_c_args[2], algorithm = 'brute')

def SpectClust(list_c_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns an Spectral Clustering object based on the list of arguements passed; 
    # this method is designed to find the most appropriate type of clustering method for the data
    #
    # Requirements:
    # package sklearn.cluster.DBSCAN
    #
    # Inputs:
    # list_c_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # list_c_args[0] -> number of clusters, <int>
    # list_c_args[1] -> affinity, <string>
    # list_c_args[2] -> eigen_solver, <string>
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Spectral Clustering based object
    # Desc: this is an object which clusters the group of data bassed to the object
    ###############################################################################################
    ###############################################################################################

    return SpectralClustering(n_clusters = list_c_args[0], affinity = list_c_args[1], eigen_solver = list_c_args[2])

def FixDbscanResults(list_dbscan_results):
    ###############################################################################################
    ###############################################################################################
    #
    # this method increments the DBSCAN results by 1 to account for the Noise points labled -1 in the results
    # this  is needed for numpy.bincount() function
    #
    # Requirements:
    # package numpy
    #
    # Inputs:
    # list_dbscan_results
    # Type: list
    # Desc: the list of parameters for the clustering object
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: list
    # Desc: this is an object which clusters the group of data bassed to the object
    ###############################################################################################
    ###############################################################################################    

    # lists
    list_return_results = list()
    
    # loop through the results from DBSCAN
    for result in list_dbscan_results:
        # set temp variables
        list_temp = list()

        # informational lists
        cluster_labels = result[0]    

        # find the -1 labels from DBSCAN cluster results
        # add 1 to help in the evaluation
        for label in cluster_labels:
            label += 1
            list_temp.append(label)

        # convert to numpy array
        array_label = numpy.array(list_temp)

        # replace the list of labels
        result.pop(0)
        result.insert(0, array_label)

        # add to return list
        list_return_results.append(result)

    # return list
    return list_return_results

def SwitchCluster(string_clust, list_args):
    ################################################################################################
    ###############################################################################################
    #
    # this method will take a string which identfies the the cluster algorithm and the list of arguements to generate
    # the the cluster object
    #
    # Requirements:
    # None
    #
    # Inputs:
    # string_clust
    # Type: string
    # Desc: the key for the dicitionary of cluster functions
    #
    # bool_descending
    # Type: boolean
    # Desc: flag to determine if the order is reversed, devault is ascending low to high
    # 
    # list_args
    # Type: list
    # Desc: the list of arguements for each cluster aglorithm
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: cluster object
    # Desc: the cluster object is the result of the fucntion call in which the function is the target of the key in the 
    #              dictionary
    ###############################################################################################
    ###############################################################################################
    dict_switcher = {
        'Aggl': AgglClust,
        'BatchKm': BatchKmClust,
        'Dbs': DbsClust, 
        'Spec': SpectClust
        }

    # get fucntion from dictionary
    func = dict_switcher.get(string_clust)
    
    # execute function
    return func(list_args)

def CreateClusterPrediction(list_num_clust, list_02, list_03, string_cluster):
    ################################################################################################
    ###############################################################################################
    #
    # the method will takes a list of agruements and the type of culster algorithm to use and calls the switcher function
    # which will create a list of clustering objects with an ID detailing the variables used
    #
    # Requirements:
    # method SwitchCluster()
    #
    # Inputs:
    # list_num_clust
    # type: list
    # desc: the number of clusters to seperate the data into
    #  
    # list_02
    # type: list
    # desc: parameter list
    #  
    # list_03
    # type: list
    # desc: parameter list
    #  
    # string_cluster
    # type: string
    # desc: the cluster algorithm to use
    #  
    # Important Info:
    # - the string which id's the cluster is in the form:
    #        "cluster algorithm uses|number of clusters|parameter 02|parameter 03"
    #
    # Return:
    # object
    # Type: list
    # Desc: the list of the algorithm and the id of the algorithm
    # list[x][0] -> type: sklearn.cluster object; the clustering algorithm object
    # list[x][1] -> type: string; the ID to include the type of clustering algorithm and each parameter seperated by a '|'
    ###############################################################################################
    ###############################################################################################

    # create lists
    list_cluster_algorithm = list()
    list_parameters = list()

    # loop through parameter lists
    for item_num_clust in list_num_clust:
        for item_02 in list_02:
            for item_03 in list_03:
                # create parmeter list and ID
                list_parameters = [item_num_clust, item_02, item_03]
                string_cluster_id = string_cluster + '|' + str(item_num_clust) + '|' + str(item_02) + '|' + str(item_03)

                # create different clustering algorithms based on the parmeter lists
                list_cluster_algorithm.append([SwitchCluster(string_cluster, list_parameters), string_cluster_id])

                # reset lists
                list_parameters = list()

    # return list of cluster algorithms & ID's
    return list_cluster_algorithm

def GetClusterResults(list_cluster_alg, list_matrices):
    ################################################################################################
    ################################################################################################
    #
    # this method fit and predicts the clusters based on the algorithms passed
    #
    # Requirements:
    # package sklearn.cluster
    #
    # Inputs:
    # list_cluster_alg
    # type: list
    # desc: the clustering algorithms
    #  
    # list_matrices
    # type: list
    # desc: the data matrices sparse, dense, dataframe
    # list_matrices[0] -> sparse matrix
    # list_matrices[1] -> dense matrix
    # list_matrices[2] -> dataframe matrix
    #
    # Important Info:
    # - in <list_cluster_algorithm> the order and the seperator is important;  this comes from the method
    #        CreateClusterPrediction(); the seperator to split the string on is the pipe "|" character not an
    #        uppercase i "I" or a lowercase L "l"
    #
    # Return:
    # object
    # Type: list
    # Desc: the list of the clustering results
    # list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
    #                or dense array
    # list[x][1] -> type: string; the cluster ID with the parameters
    ###############################################################################################
    ###############################################################################################

    # lists
    list_results = list()
    
    # get algorithm id
    list_id = list_cluster_alg[0][1].split(sep = '|')

    # get results
    # loop through list and predict results
    for i in range(0, len(list_cluster_alg)):
        # get the cluster algorithm type
        list_id = list_cluster_alg[i][1].split(sep = '|')

        # start time
        time_start = time.perf_counter()

        # the results of the cluster algorithm and ID
        # agglomerative and spectral clustering need a dense matrix
        # KNN and DBSCAN can use a sparse matrix
        if list_id[0] == 'Aggl' or list_id[0] == 'Spec':
            list_results.append([list_cluster_alg[i][0].fit_predict(list_matrices[1]), list_cluster_alg[i][1]])
        else:
            list_results.append([list_cluster_alg[i][0].fit_predict(list_matrices[0]), list_cluster_alg[i][1]])
        
        # elapsed time    
        time_total = time.perf_counter() - time_start
        print('%s clustering time (HH:MM:SS): %s' %(list_cluster_alg[i][1], time.strftime('%H:%M:%S', time.gmtime(time_total))))

    #time_start = time.perf_counter()
    #list_results = FitPredictCluster(list_cluster_alg, list_matrices)
    #time_total = time.perf_counter() - time_start
    #print('%s clustering results created for %s clustering process' %(len(list_results), list_id[0]))
    #print('%s clustering time (HH:MM:SS): %s' %(list_id[0], time.strftime('%H:%M:%S', time.gmtime(time_total))))

    # return value
    return list_results