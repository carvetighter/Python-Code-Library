#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# File / Package Import
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  
      
from SqlMethods import SqlMethods
from collections import Counter
from datetime import datetime
from datetime import timedelta
import time
import tkn_transform
import os
import pandas
import numpy
import warnings

# supress warnings
warnings.simplefilter('ignore')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Methods
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def def_Methods(list_cluster_results, array_sparse_matrix):
    '''
    
    below is an example of a good method comment
    
    -------------------------------------------------------------------------------------------#
    
    this method implements the evauluation criterea for the clusters of each clutering algorithms
    criterea:
           - 1/2 of the clusters for each result need to be:
               - the average silhouette score of the cluster needs to be higher then the silhouette score of all the clusters
                 combined
               - the standard deviation of the clusters need to be lower than the standard deviation of all the clusters
                 combined
           - silhouette value for the dataset must be greater than 0.5
    
    Requirements:
    package time
    package numpy
    package statistics
    package sklearn.metrics
    
    Inputs:
    list_cluster_results
    Type: list
    Desc: the list of parameters for the clustering object
    list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
                   or dense array
    list[x][1] -> type: string; the cluster ID with the parameters
    
    array_sparse_matrix
    Type: numpy array
    Desc: a sparse matrix of the samples used for clustering
        
    Important Info:
    None
    
    Return:
    object
    Type: list
    Desc: this of the clusters that meet the evaluation criterea
    list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
                   or dense array
    list[x][1] -> type: string; the cluster ID with the parameters
    list[x][2] -> type: float; silhouette average value for the entire set of data
    list[x][3] -> type: array; 1 dimensional array of silhouette values for each data sample
    list[x][4] -> type: list; list of lists, the cluster and the average silhoutte value for each cluster, the orders is sorted 
                       highest to lowest silhoutte value
                       list[x][4][x][0] -> int; cluster label
                       list[x][4][x][1] -> float; cluster silhoutte value
    list[x][5] -> type: list; a list that contains the cluster label and the number of samples in each cluster
                       list[x][5][x][0] -> int; cluster label
                       list[x][5][x][1] -> int; number of samples in cluster list[x][5][x][0]  
    '''
    #---------------------------------------------------------------------------------------------#
    # objects declarations
    #---------------------------------------------------------------------------------------------#

    #---------------------------------------------------------------------------------------------#
    # time declarations
    #---------------------------------------------------------------------------------------------#

    #---------------------------------------------------------------------------------------------#
    # iteration declarations (list, set, tuple, counter, dictionary)
    #---------------------------------------------------------------------------------------------#

    list_return = list()

    #---------------------------------------------------------------------------------------------#
    # variables declarations
    #---------------------------------------------------------------------------------------------#

    #---------------------------------------------------------------------------------------------#
    # db connections
    #---------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #---------------------------------------------------------------------------------------------#
    # sub-section comment
    #---------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # sectional comment
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #---------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #---------------------------------------------------------------------------------------------#

    #---------------------------------------------------------------------------------------------#
    # return value
    #---------------------------------------------------------------------------------------------#

    return list_return
    
class EmptyStringFilter(tkn_transform.TransformBase):
    ###############################################################################################
    ###############################################################################################
    # 
    # this class implements an empty string filter for the tokens in natural language processing.  
    # this class is a subclass of TransformBase oringaly developed by Wes Soloman from Saxoney Partners.  
    #
    # Requirements:
    # file tkn_transform
    # 
    # Important Info:
    # as part of a base class the TransformBase the implemenation assumes the functions calls are with both a
    # token and flags set().  In methods _reject_empty() and run() need to be called with both a tokens and 
    # tags set for the superclass and global methods to function properly.  this class can also be 
    # implemented into a pipe
    #
    # methods:
    # _reject_empty()
    # inputs:
    # tkn -> type: set; tokens to check for empty string
    # tag -> type: set; tags to account for if present, not used
    # return: boolean
    #        True: if token is empty string
    #        False: if token is not an empty string
    # desc: check the tokens to determine if there is an abbreviation and reject it
    #
    # run()
    # inputs:
    # _reject_abbrv -> type: method; checking a token if it is an empty string
    # tkns -> type: set; tokens
    # tags -> type: set; flags for a token (token with two meanings therefore two tokens are needed)
    # return: list_tokens, list_tags
    #        list_tokens -> type: list; transformed tokens
    #        list_tags -> type: list; tags for each token
    # desc: calls the global method filt_run() which will call the method _reject_empty() in this class; 
    #       which returns a token and tags list; the tokens list will be the tokens which the tokens are 
    #       not empty string; as defined by the set abbreviation in this class
    ###############################################################################################
    ###############################################################################################    

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # initialize / constructor
    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    def __init__(self, flgset = set()):
        # super initializer / constructer from TransformerBase
        super(EmptyStringFilter, self).__init__(flgset, tkn_transform.replace_run)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # method to evaluate token rejection based on punctuation
    #------------------------------------------------------------------------------------------------------------------------------------------------------#        
    def _reject_empty(self, tkn, tag=None):
        # split token from flag / tag
        rtkn, flg = self.flg_splitr.split(tkn)

        # test if token is empty string
        if rtkn == '':
            return True 
        return False

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # method to execute in global method run_pipeline or indivudual execution of class
    # filt_run() is a global method
    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    def run(self, tkns, tags=None):
        return tkn_transform.filt_run(self._reject_empty, tkns, tags)
        
def series_apply_remove_1st_num_string(m_string):
    '''    
    this method is designed to apply to a pandas series which takes out the first phrase that is
    numeric
    
    Requirements:
    None
    
    Inputs:
    m_string
    Type: string
    Desc: the string to search
        
    Important Info:
    None
    
    Return:
    variable
    Type: string
    Desc: the string without the first string that is numeric
    '''    
    #---------------------------------------------------------------------------------------------#
    # split the data into strings
    #---------------------------------------------------------------------------------------------#

    list_string_split = m_string.split(' ')
    string_return = ''

    #---------------------------------------------------------------------------------------------#
    # pull out the fist numeric string if present
    #---------------------------------------------------------------------------------------------#

    if list_string_split[0].isnumeric() == True:
        for string_word in list_string_split[1:]:
            string_return += string_word + ' '
    else:
        for string_word in list_string_split:
            string_return += string_word + ' '

    #---------------------------------------------------------------------------------------------#
    # return value
    #---------------------------------------------------------------------------------------------#
        
    return string_return
    
def filter_ones(m_counter_temp = Counter()):
    '''    
    this method is designed to apply to a pandas series which looks at a counter and removes all the
    counts that are of one except the counts that are latitude and longitidue
    
    Requirements:
    package Collections.Counter
    
    Inputs:
    m_counter_temp
    Type: Counter
    Desc: the counter of phrases
        
    Important Info:
    None
    
    Return:
    object
    Type: dictionary
    Desc: the phrase and the count
    example: {'phrase_01':3, 'phrase_02':4, ...}
    '''     
    
    #---------------------------------------------------------------------------------------------#
    # iteration declarations (list, set, tuple, counter, dictionary)
    #---------------------------------------------------------------------------------------------#    
    dict_temp = dict()
    
    #---------------------------------------------------------------------------------------------#
    # loop throgh the counter and filter out he counts that are only one except 
    # counts that were latitude and longitide counts
    #---------------------------------------------------------------------------------------------#
    
    for string_key, int_count in m_counter_temp.items():
        if int_count == 1:
            # take out negatives
            if '-' in string_key:
                string_key_01 = string_key.replace('-', '')
            else:
                string_key_01 = string_key

            # test if key is similar to 43.5656,78.45455
            if (',' in string_key_01) and len(string_key_01.split(',')) == 2:
                list_split_ll_00 = string_key_01.split(',')
                if '.' in list_split_ll_00[0] and '.' in list_split_ll_00[1]:
                    set_ll_00 = set()
                    for string_ll in list_split_ll_00:
                        list_split_ll_01 = string_ll.split('.')
                        for string_ll_01 in list_split_ll_01:
                            set_ll_00.add(string_ll_01.isnumeric())
                    if set_ll_00 != {True}:
                        dict_temp.update({string_key:int_count})
            
            # test for counts similar to 45.54543
            elif '.' in string_key_01 and len(string_key_01.split('.')) == 2:
                list_string_dec = string_key_01.split('.')
                set_dec = set()
                for string_dec in list_string_dec:
                    set_dec.add(string_dec.isnumeric())
                if set_dec != {True}:
                    dict_temp.update({string_key:int_count})
            else:
                dict_temp.update({string_key:int_count})

    #---------------------------------------------------------------------------------------------#
    # return
    #---------------------------------------------------------------------------------------------#

    return dict_temp

def replace_ll(m_list_tokens):
    '''
    this method is designed to apply to a pandas series which replaces the latidue and longitude
    with a list, tokenizes the latitiude and longitude
    
    Requirements:
    None
    
    Inputs:
    m_counter_temp
    Type: Counter
    Desc: the counter of phrases
        
    Important Info:
    None
    
    Return:
    object
    Type: list
    Desc: list of tokens with the latitude and longitide split out
    '''
    
    #---------------------------------------------------------------------------------------------#
    # loop throgh the list of tokens
    #---------------------------------------------------------------------------------------------#    
    
    for string_token in m_list_tokens:
        # take out negatives
        if '-' in string_token:
            string_key_01 = string_token.replace('-', '')
        else:
            string_key_01 = string_token
    
        # test if lat and longitude combined
        if (',' in string_key_01) and len(string_key_01.split(',')) == 2:
            list_split_ll_00 = string_key_01.split(',')
            if '.' in list_split_ll_00[0] and '.' in list_split_ll_00[1]:
                set_ll_00 = set()
                for string_ll in list_split_ll_00:
                    list_split_ll_01 = string_ll.split('.')
                    for string_ll_01 in list_split_ll_01:
                        set_ll_00.add(string_ll_01.isnumeric())
                if set_ll_00 == {True}:
                    int_index = m_list_tokens.index(string_token)
                    list_ll = string_token.split(',')
                    list_first = m_list_tokens[:int_index]
                    list_first.extend(list_ll)                    
                    if int_index < len(m_list_tokens) - 1:
                        list_first.extent(m_list_tokens[int_index + 1:])
                    return list_first

    #---------------------------------------------------------------------------------------------#
    # not lat long combined token
    #---------------------------------------------------------------------------------------------#

    return m_list_tokens
    
def create_bi_grams(m_gen_bi_gram):
    '''
    this method is designed to apply to a pandas series which takes a generator which returns
    a list of bigrams
    
    Requirements:
    package nltk.
    
    Inputs:
    m_gen_bi_gram
    Type: generator
    Desc: generator which produces the bigrams
        
    Important Info:
    None
    
    Return:
    object
    Type: list
    Desc: list of bigrams
    '''    
    #---------------------------------------------------------------------------------------------#
    # iteration declarations (list, set, tuple, counter, dictionary)
    #---------------------------------------------------------------------------------------------#  
    
    list_return = list()

    #---------------------------------------------------------------------------------------------#
    # loop through generator
    #---------------------------------------------------------------------------------------------#  

    for bi_gram in m_gen_bi_gram:
        list_return.append(bi_gram)

    #---------------------------------------------------------------------------------------------#
    # return value
    #---------------------------------------------------------------------------------------------#  

    return list_return
    
def series_elements_to_string(m_iterable):
    '''
    this method is designed for a panda series which will take a list, set, tuple, dictionary 
    (keys in this case) to a string with spaces
    
    Requirements:
    None
    
    Inputs:
    m_iterable
    Type: iterable: list, sets, tuple, dictionary
    Desc: the iterable for each element of the pandas series
        
    Important Info:
    None
    
    Return:
    variable
    Type: string
    Desc: string which represents the tokens
    '''

    #---------------------------------------------------------------------------------------------#
    # return value
    #---------------------------------------------------------------------------------------------#

    return ' '.join(map(str, m_iterable))
    
def series_elements_to_bigram_string(m_bigram_iterable):
    '''
    this method is designed for a panda series which will take a list, set, tuple, dictionary 
    (keys in this case) to a string with spaces; this is designed for bi-grams created from
    nltk.ngrams(x, 2)
    
    Requirements:
    None
    
    Inputs:
    m_bigram_iterable
    Type: iterable: list, sets, tuple, dictionary
    Desc: the iterable for each element of the pandas series, these are bi-grams
    e.g. [('a', 'b'), ('b', 'c'), ('c', 'd'), ...]
        
    Important Info:
    None
    
    Return:
    variable
    Type: string
    Desc: string which represents the bi-grams
    '''
    #---------------------------------------------------------------------------------------------#
    # variables
    #---------------------------------------------------------------------------------------------#

    string_return = ''
    
    #---------------------------------------------------------------------------------------------#
    # loop through iterable and transform to string
    #---------------------------------------------------------------------------------------------#    
    
    for temp_tuple in m_bigram_iterable:
        string_return += str(temp_tuple).replace("'", '') + ' '

    #---------------------------------------------------------------------------------------------#
    # return value
    #---------------------------------------------------------------------------------------------#

    return string_return[:-1]

def series_apply_iter_to_string(m_iter):
    '''
    this method is for a pandas series; it takes an interable and returns a string of a string of its 
    elements without any single or double quotes

    Requirements:
    None
    
    Inputs:
    m_iter
    Type: iterable; list, dictionary, set, tuple
    Desc: list which is each element in the pandas series
        
    Important Info:
    None
    
    Return:
    variable
    Type: string
    Desc: the string of elements
    '''
    return ' '.join(str(x).replace("'", '') for x in m_iter)

def dataframe_apply_create_key(m_row):
    '''
    this method is for a pandas dataframe; produces the string for the key for the dataframe that is either
    duplicated by the TPI number or the address

    Requirements:
    None
    
    Inputs:
    m_row
    Type: array
    Desc: the dummy variable for the row
        
    Important Info:
    1. axis option must equal 1
    2. the series required to country_code, postal_code, city
    
    Return:
    variable
    Type: string
    Desc: the key for the datraframe
    '''
    return "(" + m_row['country_code'] + ',' + m_row['postal_code'] + ',' + m_row['city'] + ')'