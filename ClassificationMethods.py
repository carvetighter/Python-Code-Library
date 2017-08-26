###############################################################################################
###############################################################################################
#
# This import file <ClassificationMethods.py> contains methods to cluster data
#
# Requirements:
# package numpy
# package warnings
# package time
# package sklearn.cluster
#
# Methods included:
# SwitchClassifiction()
#    - switcher function for the clustering algorithms
#
# Important Info:
# None
###############################################################################################
###############################################################################################

#------------------------------------------------------------------------------------------------------------------------------------------------------#
# package import
#------------------------------------------------------------------------------------------------------------------------------------------------------#

import warnings
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# ignore all warnings
warnings.simplefilter(action = 'ignore')

###############################################################################################
###############################################################################################
#
# Methods
#
###############################################################################################
###############################################################################################

def SwitchClassification(m_string_class = '', m_list_args = list()):
    ###############################################################################################
    ###############################################################################################
    #
    # this method is a switching function with will return a classificaiton object based on the the indentifier <m_string_class>
    #
    # Requirements:
    # None
    #
    # Inputs:
    # m_string_class
    # Type: string
    # Desc: the identifier for the classifcation algorithm
    #
    # m_list_args
    # Type: list
    # Desc: a list of arguments for the classification object, each object will be different
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: classification object
    # Desc: the classificaiton object based on the algorithm and arguements
    ###############################################################################################
    ###############################################################################################    

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists / dictionary declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dict_switcher = {
        'SDGC': class_SGDC,
        'LogReg': class_LogisticRegression,
        'NaiveBayes': class_NaiveBayes,
        'KNN': class_KNN,
        'RNN': class_RNN,
        'NCentroid': class_NearestCentroid,
        'Ridge': class_Ridge,
        'RandomForest': class_RandomForest,
        'SVC': class_SVC,
        'LinSVC': class_LinSVC,
        'NuSVC': class_NuSVC
        }

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # get the classification object
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    
    class_object = dict_switcher.get(m_string_class)                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return class_object(m_list_args)

def class_SVC(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Support Vector Machine Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.svm.SVC
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: float; 'C', penalty paramter, default is 1.0
    # m_list_args[1] -> type: string; kernal, specifies the kernal for the algorithm, default is 'rbf'
    # m_list_args[2] -> type: float; tolerance; tolerance for the stopping critera, default is 0.001
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Support Vector Machine Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return SVC(C = float(m_list_args[0]), kernel = str(m_list_args[1]), tol = float(m_list_args[2]))

def class_LinSVC(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Linear Support Vector Machine Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.svm.LInearSVC
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: float; 'C', penalty paramter, default is 1.0
    # m_list_args[1] -> type: string; penalty, specifies the norm used for penalization, 'l1' or 'l2', default is 'l2'
    # m_list_args[2] -> type: float; tolerance; tolerance for the stopping critera, default is 0.0001
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Linear Support Vector Machine Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return LinearSVC(C = float(m_list_args[0]), penalty = str(m_list_args[1]), tol = float(m_list_args[2]))

def class_NuSVC(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Nu-Support Vector Machine Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.svm.SVC
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: float; 'nu', upper bound of the fraction of training errors and a lower bound of the fraction of 
    #                                support vectors, default is 0.5
    # m_list_args[1] -> type: string; kernal, specifies the kernal for the algorithm, 'linear' 'poly' 'rbf' 'sigmoid', 'precomputed',
    #                                default is 'rbf'
    # m_list_args[2] -> type: float; tolerance; tolerance for the stopping critera, default is 0.001
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Nu-Support Vector Machine Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return NuSVC(nu = float(m_list_args[0]), kernel = str(m_list_args[1]), tol = float(m_list_args[2]))

def class_SGDC(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Stochastic Gradient Descent Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.linear_model.SGDClassifier
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: string; loss function; default is 'hinge'
    # m_list_args[1] -> type: string, penalty, default is 'l2'
    # m_list_args[2] -> type: float; alpha, the regulation term, default is 0.0001
    # m_list_args[3] -> type: int; number of passes for the training set, default is 5
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Stochastic Gradient Descent Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return SGDClassifier(loss = str(m_list_args[0]), penalty = str(m_list_args[1]), alpha = float(m_list_args[2]), 
                      n_iter = int(m_list_args[3]))

def class_LogisticRegression(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Logistic Regression Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.linear_model.LostisticRegression
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: string; penalty, 'l1' or 'l2'
    # m_list_args[1] -> type: float; 'C', inverse of regularization strength, default = 1.0, smaller value stronger regularization
    # m_list_args[2] -> type: string; solver function, for small data sets or L1 penalty use 'liblinear', multinomial loss use
    #                                'lbfgs' or 'newton-cg', large dataset use 'sag'
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Logistic Regression Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return LogisticRegression(penalty = str(m_list_args[0]), C = float(m_list_args[1]), solver = str(m_list_args[2]))

def class_NaiveBayes(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Gaussian Niave Bayes Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.naive_bayes.GaussianNB
    #
    # Inputs:
    # None
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Gaussian Niave Bayes Classifier
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return GaussianNB()

def class_KNN(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a K-Nearest Neighbors Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.neighbors.KNeighborsClassifier
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: int; # of neighbors, default is 5
    # m_list_args[1] -> type: string; algorithm, with the options 'auto', 'ball_tree', 'kd_tree', 'brute'
    # m_list_args[2] -> type: string or distance metric object; distance metric, default is 'minkowski, see 
    #                                sklearn.neighbors.DistanceMetric class in sklearn for all distance metrics
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: K-Nearest Neighbors Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return KNeighborsClassifier(n_neighbors = int(m_list_args[0]), algorithm = str(m_list_args[1]), metric = str(m_list_args[2]))

def class_RNN(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Radius Nearest Neighbors Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.neighbors.RadiusNeighborsClassifier
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: float; range of parameter space
    # m_list_args[1] -> type: string; algorithm, with the options 'auto', 'ball_tree', 'kd_tree', 'brute'
    # m_list_args[2] -> type: string or distance metric object; distance metric, default is 'minkowski, see 
    #                                sklearn.neighbors.DistanceMetric class in sklearn for all distance metrics
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Radius Nearest Neighbors Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return RadiusNeighborsClassifier(radius = float(m_list_args[0]), algorithm = str(m_list_args[1]), 
                                  metric = str(m_list_args[2]))

def class_NearestCentroid(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Nearest Centroid Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.neighbors.NearestCentroid
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: float; range of parameter space
    # m_list_args[1] -> type: string; algorithm, with the options 'auto', 'ball_tree', 'kd_tree', 'brute'
    # m_list_args[2] -> type: string or distance metric object; distance metric, default is 'minkowski, see 
    #                                sklearn.neighbors.DistanceMetric class in sklearn for all distance metrics
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Nearest Centroid Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return NearestCentroid(metric = str(m_list_args[0]))

def class_Ridge(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Ridge Regression Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.linear_model.RidgeClassifier
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: float; range of parameter space
    # m_list_args[1] -> type: string; solver algorithm, options 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'
    # m_list_args[2] -> type: float; tolerance, percision of the solution
    #                                sklearn.neighbors.DistanceMetric class in sklearn for all distance metrics
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Ridge Regression  Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return RidgeClassifier(alpha = float(m_list_args[0]), solver = str(m_list_args[1]), 
                                  tol = str(m_list_args[2]))

def class_RandomForest(m_list_args):
    ###############################################################################################
    ###############################################################################################
    #
    # this method returns a Random Forest Classifier Object based on the list of arguements passed; 
    #
    # Requirements:
    # package sklearn.ensemble.RandomForestClassifier
    #
    # Inputs:
    # m_list_args
    # Type: list
    # Desc: the list of parameters for the clustering object
    # m_list_args[0] -> type: int; number of trees in the forest, default is 10
    # m_list_args[1] -> type: string; criterion, measure to quantify the split, default is 'gini'
    # m_list_args[2] -> type: int, float, string; max_features; number of features to consider when looking for the best split,
    #                                see classifier documentaions default is 'auto'
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: Ridge Regression  Classifier Object
    # Desc: this is an object which classifies a result and data set
    ###############################################################################################
    ###############################################################################################

    return RandomForestClassifier(n_estimators = int(m_list_args[0]), criterion = str(m_list_args[1]), 
                                  max_features = m_list_args[2])