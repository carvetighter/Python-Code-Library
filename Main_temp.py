"""
This is the main script / file to determine which type of address (secondary name with street numbers, secondary 
name without street numbers, no secondary name with street numbers, no secondary name without street
numbers) produces the highest number and most accurate results.  This analysis builds on the previous analyis 
in cmdm_phase_02_de_dup_01.  This analysis will use unigrams and exclude the tokens / phrases with a 
count of one in the population.  This analysis will use a constanct cosine distance, the distance used in
cmdm_phase_02_de_dup_01 was 0.1, which is arbitrary.

The results of analysis:
??

Next phase in the analysis:
??
"""

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# File / Package Import
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#        

from SqlMethods import SqlMethods
from collections import Counter
from Timer import Timer
import os
import pandas
import numpy
import nltk
import PuncFilter
import tkn_transform
import synonym
import warnings
import cntr_util
import pickle
from cmdm_phase_02_temp_methods_01 import EmptyStringFilter
from cmdm_phase_02_temp_methods_01 import series_apply_remove_1st_num_string
from cmdm_phase_02_temp_methods_01 import filter_ones
from cmdm_phase_02_temp_methods_01 import replace_ll
from cmdm_phase_02_temp_methods_01 import create_bi_grams
from cmdm_phase_02_temp_methods_01 import series_elements_to_bigram_string
from cmdm_phase_02_temp_methods_01 import series_apply_iter_to_string
from cmdm_phase_02_temp_methods_01 import dataframe_apply_create_key
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from sklearn.metrics import pairwise
from datetime import datetime

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
    
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    
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

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # sub-section comment
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # sectional comment
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_return

def get_customer_data(m_sql_class, m_string_sql_table):
    '''    
    this method retreives the data from the Cordy's database and conducts some column filtering; returns
    two pandas dataframes; deduplicated by TPI and duplicates by TPI
    
    Requirements:
    package pandas
    file SqlMethods
    
    Inputs:
    m_sql_class
    Type: sql class
    Desc: the sql class connected to the InfoShareStage 
    
    m_string_sql_table
    Type: string
    Desc: the sql table to pull the data
        
    Important Info:
    None
    
    Return:
    objects, two dataframes
    Type: pandas dataframes
    Desc: the dataframes with the customer data; first dataframe is no duplicates as identified by the TPI
              and the second dataframe is the duplicates identified by the TPI number; the columns are the same
    dataframes['DB_NBR'] -> type: string; ??
    dataframes['CUSTOMER_NAME'] -> type: string; ??
    dataframes['SECONDARY_NAME'] -> type: string; ??
    dataframes['ADDRESS_1'] -> type: string; ??
    dataframes['ADDRESS_2'] -> type: string; ??
    dataframes['CITY'] -> type: string; ??
    dataframes['STATE'] -> type: string; ??
    dataframes['POSTAL_CODE'] -> type: string; ??
    dataframes['COUNTRY_CODE'] -> type: string; ??
    dataframes['COUNTRY'] -> type: string; ??
    dataframes['MAIL_ADDRESS_1'] -> type: string; ??
    dataframes['MAIL_ADDRESS_2'] -> type: string; ??
    dataframes['MAIL_CITY'] -> type: string; ??
    dataframes['MAIL_STATE'] -> type: string; ??
    dataframes['MAIL_POSTAL_CODE'] -> type: string; ??
    dataframes['PARETN_DB_NBR'] -> type: string; ??
    dataframes['SUPERSEDED_BY_DB_NBR'] -> type: string; ??
    dataframes['TYPE_CODE'] -> type: string; ??
    dataframes['CUSTOME_TYPE'] -> type: string; ??
    dataframes['GLOBAL_ACCOUNT_NAME'] -> type: string; ??
    dataframes['ACCOUNT_TYPE'] -> type: string; ??
    dataframes['ALLIANCE_PARTNER'] -> type: string; ??
    dataframes['INDUSTRY_CODE'] -> type: string; ??
    dataframes['INDUSTRY'] -> type: string; ??
    dataframes['INDUSTRY_GRP'] -> type: string; ??
    dataframes['MARKET'] -> type: string; ??
    dataframes['ACTIVE'] -> type: string; ??
    dataframes['VERIFIED'] -> type: string; ??
    dataframes['ACCOUNT_NUMBER'] -> type: string; ??
    dataframes['ACCOUNT_NAME'] -> type: string; ??
    dataframes['GEO_CODE'] -> type: string; ??
    dataframes['IS_PRIMARY'] -> type: string; ??
    dataframes['SOURCE'] -> type: string; ??
    dataframes['DIVISION'] -> type: string; ??
    dataframes['SE_ID'] -> type: string; ??
    dataframes['SE_CODE'] -> type: string; ??
    dataframes['SE_NAME'] -> type: string; ??
    dataframes['SE_ROLE'] -> type: string; ??
    dataframes['SE_FUNCTION'] -> type: string; ??
    dataframes['SLS_DISTRICT_CODE'] -> type: string; ??
    dataframes['SLS_DISTRICT'] -> type: string; ??
    dataframes['ETLInsertTS'] -> type: string; ??
    dataframes['ETLBatchID'] -> type: string; ??
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    string_query_where = "SESTATENAME = 'Waiting For Data Changes'"
    string_query = m_sql_class.gen_select_statement(m_string_select = '*',
                                                                                     m_string_from = m_string_sql_table,
                                                                                     m_string_where = string_query_where)

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # pull data from Cordys
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_query = m_sql_class.query_select(string_query)
    list_columns = m_sql_class.get_table_columns(m_string_sql_table)
    if list_columns[0] == True and list_query[0] == True:
        dataframe_cordys = pandas.DataFrame(data = list_query[1], columns = list_columns[1])

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # find duplictes on the TPI and fill the potential columns with empty string
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'dataframe_cordys' in locals():
        # id duplicates
        dataframe_dup_db_nbr = dataframe_cordys[dataframe_cordys.duplicated(
                                                                        subset = ['DB_NBR'], 
                                                                        keep = False)]
        dataframe_dedup_db_nbr = dataframe_cordys.drop_duplicates(['DB_NBR'])
        del dataframe_cordys

        # reindex dataframes
        dataframe_dedup_db_nbr.reset_index(inplace = True)
        dataframe_dup_db_nbr.reset_index(inplace = True)

        # fill nulls with empty string
        list_series_names = ['PARTNER_NAME', 'SECONDARY_NAME', 'ADDRESS_1',
                                          'STATE', 'POSTAL_CODE', 'COUNTRY_CODE', 'GEO_CODE']
        for string_series in list_series_names:
            # deduplicated dataframe
            array_dedup = dataframe_dedup_db_nbr[string_series].isnull()
            dataframe_dedup_db_nbr[string_series][array_dedup] = ''

            # duplicates dataframe
            array_dup = dataframe_dup_db_nbr[string_series].isnull()
            dataframe_dup_db_nbr[string_series][array_dup] = ''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    
    set_return = set()
    for string_df in ['dataframe_dedup_db_nbr', 'dataframe_dup_db_nbr']:
        if string_df in locals():
            set_return.add(True)
        else:
            set_return.add(False)

    if set_return == {True}:
        return dataframe_dedup_db_nbr, dataframe_dup_db_nbr
    else:
        return None, None

def build_addresses_to_dedup(m_dataframe):
    '''
    this method takes the dataframe that are deduplicated by the TPI number, creates a dataframe with four
    new address Series to deduplicate and identifies duplicate addresses by the most restricitve address
    Series
    
    Requirements:
    package pandas
    file cmdm_phase_02_temp_methods
    
    Inputs:
    m_dataframe
    Type: pandas dataframe
    Desc: the deduplicated dataframe from the TPI number

    Important Info:
    the indexes of the dataframes returned relate to the dataframe passed, in this case it's the dataframe 
    deduplicated by the TPI number
    m_dataframe['DB_NBR'] -> type: string; ??
    m_dataframe['CUSTOMER_NAME'] -> type: string; ??
    m_dataframe['SECONDARY_NAME'] -> type: string; ??
    m_dataframe['ADDRESS_1'] -> type: string; ??
    m_dataframe['ADDRESS_2'] -> type: string; ??
    m_dataframe['CITY'] -> type: string; ??
    m_dataframe['STATE'] -> type: string; ??
    m_dataframe['POSTAL_CODE'] -> type: string; ??
    m_dataframe['COUNTRY_CODE'] -> type: string; ??
    m_dataframe['COUNTRY'] -> type: string; ??
    m_dataframe['MAIL_ADDRESS_1'] -> type: string; ??
    m_dataframe['MAIL_ADDRESS_2'] -> type: string; ??
    m_dataframe['MAIL_CITY'] -> type: string; ??
    m_dataframe['MAIL_STATE'] -> type: string; ??
    m_dataframe['MAIL_POSTAL_CODE'] -> type: string; ??
    m_dataframe['PARETN_DB_NBR'] -> type: string; ??
    m_dataframe['SUPERSEDED_BY_DB_NBR'] -> type: string; ??
    m_dataframe['TYPE_CODE'] -> type: string; ??
    m_dataframe['CUSTOME_TYPE'] -> type: string; ??
    m_dataframe['GLOBAL_ACCOUNT_NAME'] -> type: string; ??
    m_dataframe['ACCOUNT_TYPE'] -> type: string; ??
    m_dataframe['ALLIANCE_PARTNER'] -> type: string; ??
    m_dataframe['INDUSTRY_CODE'] -> type: string; ??
    m_dataframe['INDUSTRY'] -> type: string; ??
    m_dataframe['INDUSTRY_GRP'] -> type: string; ??
    m_dataframe['MARKET'] -> type: string; ??
    m_dataframe['ACTIVE'] -> type: string; ??
    m_dataframe['VERIFIED'] -> type: string; ??
    m_dataframe['ACCOUNT_NUMBER'] -> type: string; ??
    m_dataframe['ACCOUNT_NAME'] -> type: string; ??
    m_dataframe['GEO_CODE'] -> type: string; ??
    m_dataframe['IS_PRIMARY'] -> type: string; ??
    m_dataframe['SOURCE'] -> type: string; ??
    m_dataframe['DIVISION'] -> type: string; ??
    m_dataframe['SE_ID'] -> type: string; ??
    m_dataframe['SE_CODE'] -> type: string; ??
    m_dataframe['SE_NAME'] -> type: string; ??
    m_dataframe['SE_ROLE'] -> type: string; ??
    m_dataframe['SE_FUNCTION'] -> type: string; ??
    m_dataframe['SLS_DISTRICT_CODE'] -> type: string; ??
    m_dataframe['SLS_DISTRICT'] -> type: string; ??
    m_dataframe['ETLInsertTS'] -> type: string; ??
    m_dataframe['ETLBatchID'] -> type: string; ??
    
    Return:
    objects
    Type: pandas dataframes
    Desc: the first dataframe with the addresses to deduplicate; the second dataframe with the records that
              are identified as duplicates with the new address
    dataframes['tpi'] -> type: string; tpi number which is the customer identifier
    dataframes['country_code'] -> type: string; country code from the Cordys database
    dataframes['postal_code'] -> type: string; postal code from the Cordys database
    dataframes['city'] -> type: string; city from the Cordys database
    dataframes['address_to_dedup_00'] -> type: pandas Series; customer name, no street numbers
    dataframes['address_to_dedup_01'] -> type: pandas Series; customer name, secondary name, 
                                                                    no street numbers
    dataframes['address_to_dedup_02'] -> type: pandas Series; customer name, street numbers
    dataframes['address_to_dedup_03'] -> type: pandas Series; customer name, scondary name, street 
                                                                    numbers
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_addresses = pandas.DataFrame()
    series_address_no_street_num = m_dataframe.ADDRESS_1.apply(
                                                                series_apply_remove_1st_num_string)
    series_address_no_street_num.name = 'address_new_00'

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_columns = ['tpi', 'country_code', 'postal_code', 'city', 'address_to_dedup_00', 'address_to_dedup_01', 
                            'address_to_dedup_02', 'address_to_dedup_03']

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # customer name, no street numbers
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_addresses['address_to_dedup_00'] = \
            m_dataframe.PARTNER_NAME.str.strip() + ' ' + \
            series_address_no_street_num.str.strip() + ' ' + \
            m_dataframe.STATE.str.strip() + ' ' + \
            m_dataframe.POSTAL_CODE.str.strip() + ' ' + \
            m_dataframe.COUNTRY_CODE.str.strip() + ' ' + \
            m_dataframe.GEO_CODE.str.strip()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # customer name, secondary name, no street numbers
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_addresses['address_to_dedup_01'] = \
        m_dataframe.PARTNER_NAME.str.strip() + ' ' + \
        m_dataframe.SECONDARY_NAME.str.strip() + ' ' + \
        series_address_no_street_num.str.strip() + ' ' + \
        m_dataframe.STATE.str.strip() + ' ' + \
        m_dataframe.POSTAL_CODE.str.strip() + ' ' + \
        m_dataframe.COUNTRY_CODE.str.strip() + ' ' + \
        m_dataframe.GEO_CODE.str.strip()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # customer name, street numbers
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_addresses['address_to_dedup_02'] = \
        m_dataframe.PARTNER_NAME.str.strip() + ' ' + \
        m_dataframe.ADDRESS_1.str.strip() + ' ' + \
        m_dataframe.STATE.str.strip() + ' ' + \
        m_dataframe.POSTAL_CODE.str.strip() + ' ' + \
        m_dataframe.COUNTRY_CODE.str.strip() + ' ' + \
        m_dataframe.GEO_CODE.str.strip()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # customer name, scondary name, street numbers
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_addresses['address_to_dedup_03'] = \
        m_dataframe.PARTNER_NAME.str.strip() + ' ' + \
        m_dataframe.SECONDARY_NAME.str.strip() + ' ' + \
        m_dataframe.ADDRESS_1.str.strip() + ' ' + \
        m_dataframe.STATE.str.strip() + ' ' + \
        m_dataframe.POSTAL_CODE.str.strip() + ' ' + \
        m_dataframe.COUNTRY_CODE.str.strip() + ' ' + \
        m_dataframe.GEO_CODE.str.strip()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # take out nulls and add city, country code, postal code
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    array_bool = dataframe_addresses.isnull()
    dataframe_addresses[array_bool] = ''
    del array_bool

    dataframe_addresses['city'] = m_dataframe.CITY
    dataframe_addresses['country_code'] = m_dataframe.COUNTRY_CODE
    dataframe_addresses['postal_code'] = m_dataframe.POSTAL_CODE
    dataframe_addresses['tpi'] = m_dataframe.DB_NBR
    dataframe_addresses.index = m_dataframe.index

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # find duplicates and deduplicate based on address_to_dedup_03
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_dedup_addresses = dataframe_addresses.drop_duplicates(['address_to_dedup_02'])
    dataframe_dup_addresses = dataframe_addresses[dataframe_addresses.duplicated(
                                                        subset = ['address_to_dedup_02'],
                                                        keep = False)]
    del dataframe_addresses

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return dataframe_dedup_addresses[list_columns], dataframe_dup_addresses[list_columns]

def ct_read_all_tsv(m_string_tsv):
    '''
    this method returns a string of all the lines in the tsv file
    
    Requirements:
    None
    
    Inputs:
    m_string_tsv
    Type: string
    Desc: the file name of the .tsv file
        
    Important Info:
    The .tsv file needs to be in the same directory as this python script file
    
    Return:
    variable
    Type: string
    Desc: the contents of the .tsv file
    '''

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    
    string_return = ''
    with open(m_string_tsv, 'r') as file:
        string_return = file.read()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return string_return

def create_tokens(m_dataframe, m_bool_take_out_ones):
    '''
    this method creates the tokens (uni-grams & bi-grams) for the dataframe
    
    Requirements:
    package pandas
    file cmdm_phase_02_temp_methods
    
    Inputs:
    m_dataframe
    Type: pandas dataframe
    Desc: the dataframe of addresses to build the tokens from
    m_dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
    m_dataframe['country_code'] -> type: string; country code from the Cordys database
    m_dataframe['postal_code'] -> type: string; postal code from the Cordys database
    m_dataframe['city'] -> type: string; city from the Cordys database
    m_dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
    m_dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
    m_dataframe['address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                    numbers

    m_bool_take_out_ones
    Type: boolean
    Desc: the boolean to load the tokens that are only have a count of one in the population; this 
    determined through a previous analysis

    Important Info:
    None
    
    Return:
    object
    Type: pandas dataframe
    Desc: this of the clusters that meet the evaluation criterea
    dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
    dataframe['country_code'] -> type: string; country code from the Cordys database
    dataframe['postal_code'] -> type: string; postal code from the Cordys database
    dataframe['city'] -> type: string; city from the Cordys database
    dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
    dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
    dataframe['address_to_dedup_03'] -> type: sgtring; customer name, scondary name, street 
                                                                    numbers
    dataframe['tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
    dataframe['tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
    dataframe['tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
    dataframe['tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    # copy of dataframe
    dataframe_temp = m_dataframe.copy()              

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # tokenize the addresses
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    for string_series in dataframe_temp:
        if string_series[:-3] == 'address_to_dedup':
            # create column name
            string_tokens = 'tokens_' + string_series[-2:]

            # create_tokens
            dataframe_temp[string_tokens] = dataframe_temp[string_series].apply(
                                                                lambda x: nltk.tokenize.word_tokenize(x))

            # replace latitude and longitude token that was not split
            dataframe_temp[string_tokens] = dataframe_temp[string_tokens].apply(
                                                                    lambda x: replace_ll(x))
    
    # variable cleanup
    del string_series, string_tokens

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # combine the *.tsv files 
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    string_comb = ct_read_all_tsv('customer_synonym.tsv')
    string_comb += ct_read_all_tsv('ones.tsv')

    with open('synonym.tsv', 'w') as syn_file:
        syn_file.write(string_comb)
    del string_comb

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # define token transform pipe and match generalizer
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    # create match generalizer
    match_gen = synonym.MatchGeneralizer()

    # tokens / phrases that are of count = 1 and not a latitude / longitude coordinate
    if m_bool_take_out_ones == True:
        match_gen = match_gen.load_from_file('synonym.tsv')

    # add some additional phrases
    match_gen.add(('inc.',), ('',))
    match_gen.add(('inc',), ('',))
    match_gen.add(('llc',), ('',))
    match_gen.add(('llp',), ('',))
    match_gen.add(('cith',), ('city',))
    match_gen.add(('co.',), ('',))
    match_gen.add(('co',), ('',))
    match_gen.add(('company',), ('',))
    match_gen.add(('corporation',), ('',))
    match_gen.add(('ltd.',), ('',))

    # define tokens transformation pipeline
    tkn_trans_pipe = [PuncFilter.PuncFilter(),
                  PuncFilter.AbbrvFilter(),
                  tkn_transform.LowerCaseReplacer(),
                  tkn_transform.StopWordFilter(nltk.corpus.stopwords.words('english')),
                  tkn_transform.MatchGeneralizeTransformer(match_gen),
                  EmptyStringFilter()]

    # run pipeline on each pandas series of tokens
    for string_series in dataframe_temp:
        if string_series[:-3] == 'tokens':
            dataframe_temp[string_series] = dataframe_temp[string_series].apply(
                                                                    lambda x: tkn_transform.run_pipeline(tkn_trans_pipe, x)[0])

    # object / variable cleanup
    del string_series, tkn_trans_pipe, match_gen

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # create bi-grams; commented out because focusing only on uni-grams
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #for string_series in dataframe_temp:
    #    if string_series[:-3] == 'tokens':
    #        # create column name
    #        string_bigrams = 'bi-grams_' + string_series[-2:]

    #        # create bi-grams
    #        dataframe_temp[string_bigrams] = dataframe_temp[string_series].apply(
    #                                                                    lambda x: create_bi_grams(nltk.ngrams(x, 2)))

    # variable clean-up
    #del string_series, string_bigrams

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return dataframe_temp

def ca_tfidf_matrix(m_list_grams):
    '''
    this method creates the tfidf counts for each pandas series in the list
    
    Requirements:
    package sklearn.feature_extraction.text.TfidfVectorizer
    package pandas
    
    Inputs:
    m_list_grams
    Type: list
    Desc: the list of parameters for the clustering object
    list[x] -> type: pandas Series; the strings to create the tfidf creates

    Important Info:
    None
    
    Return:
    object
    Type: list
    Desc: this of the clusters that meet the evaluation criterea
    list[x] -> type: tfidf dictionary, the tfidf calculation for each entry in the series
    eg list[0] = {'phrase_00':0.456, 'phrase_01':0.764, ...}
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_temp = list()              

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # create tfidf matricies
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    for series_gram in m_list_grams:
        tfidf_vect = TfidfVectorizer()
        list_temp.append(tfidf_vect.fit_transform(series_gram))                 

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'series_gram' in locals():
        del series_gram
    if 'tfidf_vect' in locals():
        del tfidf_vect

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_temp

def ca_dist_lcl(m_list_tfidf, m_string_dist):
    '''
    this method creates the distance matrix for the clustering algorithm and the lower control limit
    which will determine 
    
    Requirements:
    package scipy.spatial.distance
    
    Inputs:
    m_list_tfidf
    Type: list
    Desc: the list of parameters for the clustering object
    list[x] -> type: tfidf dictionary; tfidf calculations for each phrase in the document

    m_string_dist
    Type: string
    Desc: distance metric for the calculation

    Important Info:
    None
    
    Return:
    object
    Type: lists
    Desc: list of distance matrix for each entry of the tfidf values and a list for the lower control limit
                for each entry in the list passed in m_list_tfidf
    list_dist_matrix[x] -> type: numpy array; the distance matrix
    list_lcl[x] -> type: float; the lower control limit which will determine if there is a duplicate in the group
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_dist_matrix = list()
    list_lcl = list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # create distance matricies and lower control limits
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    # cosine distance
    for tfidf_matrix in m_list_tfidf:
        dist_calc = distance.pdist(tfidf_matrix.toarray(), metric = m_string_dist)
        float_lcl_00 = dist_calc.mean() - (dist_calc.std() * 1.880)
        float_lcl_01 = dist_calc.mean() - dist_calc.std()
        float_lcl_02 = 0.1 # this is an assumption based on the concept of the cosine distance
        #if float_lcl_00 > 0.:
        #    float_lcl = float_lcl_00
        ##elif float_lcl_01 > 0:
        ##    float_lcl = float_lcl_01
        #else:
        #    float_lcl = float_lcl_02
        #list_lcl.append(float_lcl)
        list_lcl.append(float_lcl_02)
        list_dist_matrix.append(distance.squareform(dist_calc))

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'tfidf_matrix' in locals():
        del tfidf_matrix
    if 'dist_calc' in locals():
        del dist_calc
    if 'float_lcl' in locals():
        del float_lcl

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_dist_matrix, list_lcl

def ca_cluster(m_list_dist, m_list_lcl):
    '''
    this method clusters the distance matricies based on the lower control limit
    
    Requirements:
    package sklearn.cluster.DBSCAN
    
    Inputs:
    m_list_dist    
    Type: list
    Desc: distance matricies for each group
    list[x] -> type: numpy array; distance matricies

    m_list_lcl    
    Type: list
    Desc: the lower control limit to determine the potential duplicate
    list[x] -> type: float; 

    Important Info:
    None
    
    Return:
    object
    Type: lists
    Desc: list of distance matrix for each entry of the tfidf values and a list for the lower control limit
                for each entry in the list passed in m_list_tfidf
    list_dist_matrix[x] -> type: numpy array; the distance matrix
    list_lcl[x] -> type: float; the lower control limit which will determine if there is a duplicate in the group
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_temp = list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # cluster results
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if len(m_list_dist) == len(m_list_lcl):
        for int_index in range(0, len(m_list_dist)):
                    list_temp.append(DBSCAN(min_samples = 2, eps = m_list_lcl[int_index], metric = 'precomputed').\
                                                    fit(m_list_dist[int_index]))              

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'int_index' in locals():
        del int_index

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_temp

def ca_check_cluster(m_list_cluster, m_bool_ones, m_string_gram_label, m_key, m_dataframe):
    '''
    this method checks the cluster results for duplicates
    
    Requirements:
    package sklearn.cluster.DBSCAN
    package pandas
    
    Inputs:
    m_list_cluster    
    Type: list
    Desc: density cluster objects
    list[x] -> type: dbscan cluster object; the results of the cluster results

    m_bool_ones    
    Type: boolean
    Desc: the flag if the tokens that had a count of one were included in the analysis

    m_string_gram_label    
    Type: string
    Desc: if the analysis is for uni-grams or bi-grams

    m_key    
    Type: tuple
    Desc: the key from the groupby index

    m_dataframe    
    Type: pandas dataframe
    Desc: the dataframe of the group
    m_dataframe['address_to_dedup_00'] -> type: pandas Series; customer name, no street numbers
    m_dataframe['address_to_dedup_01'] -> type: pandas Series; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['address_to_dedup_02'] -> type: pandas Series; customer name, street numbers
    m_dataframe['address_to_dedup_03'] -> type: pandas Series; customer name, scondary name, street 
                                                                    numbers
    m_dataframe['city'] -> type: pandas Series; city from the Cordys database
    m_dataframe['country_code'] -> type: pandas Series; country code from the Cordys database
    m_dataframe['postal_code'] -> type: pandas Series; country code from the Cordys database
    m_dataframe['tpi'] -> type: pandas Series; the customer indentification number
    m_dataframe['tokens_00'] -> type: pandas Series; uni-grams from address_to_dedup_00
    m_dataframe['tokens_01'] -> type: pandas Series; uni-grams from address_to_dedup_01
    m_dataframe['tokens_02'] -> type: pandas Series; uni-grams from address_to_dedup_02
    m_dataframe['tokens_03'] -> type: pandas Series; uni-grams from address_to_dedup_03

    Important Info:
    None
    
    Return:
    objects
    Type: list(), list()
    Desc: the list of duplicate duplicates and non-duplicates
    m_list[x][0] -> type: tuple; key for the group
    m_list[x][1] -> type: string; the label of the gram
        uni-gram -> identifies record as part of the uni-gram analysis
        bi-gram -> identifies record as part of the bi-gram analysis
        only_one_in_group -> record was not analysed because it was only one record
    m_list[x][2] -> type: string; flag to indicate if the tokens with only one count are included
        ones -> indicates records analyzed included tokens with a count of one
        non-ones -> indicates records analysed did not included tokesn with a count of one
    m_list[x][3] -> type: pandas dataframe; the dataframe with the data; format is the same
                            as m_dataframe; either duplicates or non-duplicates
    '''
    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # iterables; lists, dictionaries, sets, tuples
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_return_duplicates, list_return_nonduplicates = list(), list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # check cluster results
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    for cluster_results in m_list_cluster:
        if 0 in cluster_results.labels_:
            array_bool = cluster_results.labels_ >= 0
            if m_bool_ones == True:
                list_return_duplicates.append([m_key, m_string_gram_label, 'ones', m_dataframe[array_bool]])
                list_return_nonduplicates.append([m_key, m_string_gram_label, 'ones', m_dataframe[~array_bool]])
            else:
                list_return_duplicates.append([m_key, m_string_gram_label, 'no-ones', m_dataframe[array_bool]])
                list_return_nonduplicates.append([m_key, m_string_gram_label, 'no-ones', m_dataframe[~array_bool]])
        else:
            if m_bool_ones == True:
                list_return_nonduplicates.append([m_key, m_string_gram_label, 'ones', m_dataframe])
            else:
                list_return_nonduplicates.append([m_key, m_string_gram_label, 'no-ones', m_dataframe])

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'cluster_results' in locals():
        del cluster_results

    if 'array_bool' in locals():
        del array_bool

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_return_duplicates, list_return_nonduplicates

def cluster_addresses(m_dataframe, m_bool_ones):
    '''
    this method conducts the cluster analysis of the four different address configurations
    
    Requirements:
    package pandas
    package scipy
    package numpy
    
    Inputs:
    m_dataframe
    Type: pandas dataframe
    Desc: the dataframe to begin the cluster of the addresses
    m_dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
    m_dataframe['country_code'] -> type: string; country code from the Cordys database
    m_dataframe['postal_code'] -> type: string; postal code from the Cordys database
    m_dataframe['city'] -> type: string; city from the Cordys database
    m_dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
    m_dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
    m_dataframe['address_to_dedup_03'] -> type: sgtring; customer name, scondary name, street 
                                                                    numbers
    m_dataframe['tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
    m_dataframe['tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
    m_dataframe['tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
    m_dataframe['tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03

    m_bool_ones
    Type: boolean
    Desc: flag if the tokens with only one count were included in the analysis

    Important Info:
    the index of the dataframe returned is important; the index references the original dataframe analyzed
    dataframe_dedup_addr from the main method; those records should be referenced and inserted into
    the database for analysis
    
    Return:
    object
    Type: list of lists
    Desc: will return two lists which encompus two analysis; duplicates and non-duplicates
    list[x][0] -> type: tuple; groupby index key
    list[x][1] -> type: string; label for the type of analysis, there are three options
        uni-gram -> identifies record as part of the uni-gram analysis
        bi-gram -> identifies record as part of the bi-gram analysis
        only_one_in_group -> record was not analysed because it was only one record
    list[x][2] -> type: string; flag which indicates if there tokens that with only one count were included in the analysis;
                        two options
        ones -> indicates records analyzed included tokens with a count of one
        non-ones -> indicates records analysed did not included tokesn with a count of one
    list[x][3] -> type: pandas dataframe; the dataframe with the pertinate records; the format is the same as
                    m_dataframe
        dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
        dataframe['country_code'] -> type: string; country code from the Cordys database
        dataframe['postal_code'] -> type: string; postal code from the Cordys database
        dataframe['city'] -> type: string; city from the Cordys database
        dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
        dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                        no street numbers
        dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
        dataframe['address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                        numbers
        dataframe['tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
        dataframe['tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
        dataframe['tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
        dataframe['tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_duplicates, list_non_duplicates = list(), list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # group dataframe by the country code, postal code, city
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    groupby_index = m_dataframe.groupby(by = ['country_code', 'postal_code', 'city'])
    #groupsby_index = m_dataframe.grouby(by = ['country_code', 'postal_code'])

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # loop through groups, groupby_index is a generater and will only allow one loop through the groups
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    for key, dataframe_group in groupby_index:
        # check for more than one record in group
        if len(dataframe_group) > 1:

            #------------------------------------------------------------------------------------------------------------------------------------------------------#
            # convert tokens to list of strings
            #------------------------------------------------------------------------------------------------------------------------------------------------------#

            list_series_unigrams = list()
            #list_series_bigrams = list()

            for string_series in dataframe_group:
                if string_series[:-3] == 'tokens':
                    list_series_unigrams.append(dataframe_group[string_series].apply(
                                                                    lambda x: series_apply_iter_to_string(x)))
                #elif string_series[:-3] == 'bi_grams':
                #    list_series_bigrams.append(dataframe_group[string_series].apply(
                #                                                    lambda x: series_elements_to_bigram_string(x)))
                else:
                    pass

            # variable cleanup
            del string_series

            #------------------------------------------------------------------------------------------------------------------------------------------------------#
            # create TFIDF matrix for uni-grams and bi-grams
            #------------------------------------------------------------------------------------------------------------------------------------------------------#

            # tfidf for uni-grams
            list_tfidf_unigrams = ca_tfidf_matrix(list_series_unigrams)
            #list_tfidf_bigrams = ca_tfidf_matrix(list_series_bigrams)

            # variable / object clean-up
            #del list_series_unigrams, list_series_bigrams
            del list_series_unigrams

            #------------------------------------------------------------------------------------------------------------------------------------------------------#
            # create distance matrix, use cosine distance and lower control limit (lcl) which is the metric to identify
            # potential duplicates
            #------------------------------------------------------------------------------------------------------------------------------------------------------#

            list_dist_uni_cosine, list_lcl_unigram = ca_dist_lcl(list_tfidf_unigrams, 'cosine')
            #list_dist_uni_cosine, list_lcl_unigram = ca_dist_lcl(list_tfidf_unigrams, 'euclidean')
            #list_dist_bi_cosine, list_lcl_bigram = ca_dist_lcl(list_tfidf_bigrams, 'cosine')

            # variable / object clean-up
            #del list_tfidf_unigrams, list_tfidf_bigrams
            del list_tfidf_unigrams

            #------------------------------------------------------------------------------------------------------------------------------------------------------#
            # cluster results 
            #------------------------------------------------------------------------------------------------------------------------------------------------------#

            list_cluster_unigrams = ca_cluster(list_dist_uni_cosine, list_lcl_unigram)
            #list_cluster_bigrams = ca_cluster(list_dist_bi_cosine, list_lcl_bigram)
            
            # variable / object cleanup
            #del list_dist_uni_cosine, list_lcl_unigram, list_dist_bi_cosine, list_lcl_bigram
            del list_dist_uni_cosine, list_lcl_unigram
            
            #------------------------------------------------------------------------------------------------------------------------------------------------------#
            # check cluster results 
            #------------------------------------------------------------------------------------------------------------------------------------------------------#
            
            list_duplicates_temp, list_non_duplicates_temp = ca_check_cluster(list_cluster_unigrams, m_bool_ones, 
                                                                        'uni-grams', key, dataframe_group)
            list_duplicates.extend(list_duplicates_temp)
            list_non_duplicates.extend(list_non_duplicates_temp)
            del list_duplicates_temp, list_non_duplicates_temp

            #ca_check_cluster(list_cluster_bigrams, m_bool_ones, 'bi-grams', key, dataframe_group, 
            #                                list_duplicates, list_non_duplicates)

            # variable / object clean-up
            #del list_cluster_bigrams, list_cluster_unigrams
            del list_cluster_unigrams

        #------------------------------------------------------------------------------------------------------------------------------------------------------#
        # if there is only one record in the group
        #------------------------------------------------------------------------------------------------------------------------------------------------------#
        else:
            if m_bool_ones == True:
                list_non_duplicates.append([key,'only_one_in_group',  'ones', dataframe_group])
            else:
                list_non_duplicates.append([key,'only_one_in_group',  'no_ones', dataframe_group])
                
    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'dataframe_group' in locals():
        del dataframe_group
    if 'group_index' in locals():
        del groupby_index

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_duplicates, list_non_duplicates

def convert_to_dataframe(m_list_cluster_results):
    '''
    this method takes the the list of results and coverts them to a dataframe
    
    Requirements:
    package pandas
    
    Inputs:
    m_list_cluster_results
    Type: list
    Desc: the list of parameters for the clustering object
    list[x][0] -> type: tuple; groupby index key
    list[x][1] -> type: string; label for the type of analysis, there are three options
        uni-gram -> identifies record as part of the uni-gram analysis
        bi-gram -> identifies record as part of the bi-gram analysis
        only_one_in_group -> record was not analysed because it was only one record
    list[x][2] -> type: string; flag which indicates if there tokens that with only one count were included in the analysis;
                        two options
        ones -> indicates records analyzed included tokens with a count of one
        non-ones -> indicates records analysed did not included tokesn with a count of one
    list[x][3] -> type: pandas dataframe; the dataframe with the pertinate records; the format is the same as
                    m_dataframe
        dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
        dataframe['country_code'] -> type: string; country code from the Cordys database
        dataframe['postal_code'] -> type: string; postal code from the Cordys database
        dataframe['city'] -> type: string; city from the Cordys database
        dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
        dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                        no street numbers
        dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
        dataframe['address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                        numbers
        dataframe['tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
        dataframe['tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
        dataframe['tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
        dataframe['tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
    
    Important Info:
    None
    
    Return:
    object
    Type: pandas DataFrame
    Desc: the dataframe with the cluster results
    m_dataframe['tuple_key'] -> type: pandas Series; customer name, no street numbers
    m_dataframe['string_gram'] -> type: pandas Series; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['string_ones'] -> type: pandas Series; customer name, street numbers
    m_dataframe['string_tpi'] -> type: string; tpi number which is the customer idetnifier
    m_dataframe['string_country_code'] -> type: string; country code from the Cordys database
    m_dataframe['string_postal_code'] -> type: string; postal code from the Cordys database
    m_dataframe['string_city'] -> type: string; city from the Cordys database
    m_dataframe['string_address_to_dedup_00'] -> type: string; customer name, no street numbers
    m_dataframe['string_address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['string_address_to_dedup_02'] -> type: string; customer name, street numbers
    m_dataframe['string_address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                    numbers
    m_dataframe['list_tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
    m_dataframe['list_tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
    m_dataframe['list_tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
    m_dataframe['list_tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_data = list()
    list_columns = ['tuple_key', 'string_gram', 'string_ones', 'string_tpi', 'string_country_code', 'string_postal_code', 
                            'string_city', 'string_address_to_dedup_00', 'string_address_to_dedup_01', 
                            'string_address_to_dedup_02', 'string_address_to_dedup_03', 'list_tokens_00', 'list_tokens_01', 
                            'list_tokens_02', 'list_tokens_03']

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

    for list_result in m_list_cluster_results:
        list_init = list_result[:3]
        array_data = list_result[3].values
        
        for array_record in array_data:
            list_temp = list()
            list_temp.extend(list_init)
            list_temp.extend(array_record)
            list_data.append(list_temp)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return pandas.DataFrame(data = list_data, columns = list_columns)

def clean_dataframe(m_dataframe):
    '''
    this method cleans the key and the dataframe tokens and bi-gram series to be able to 
    insert into the sql server
    
    Requirements:
    package pandas
    
    Inputs:
    m_dataframe
    Type: pandas DataFrame
    Desc: the dataframe of cluster results
    m_dataframe['tuple_key'] -> type: tuple; customer name, no street numbers
    m_dataframe['string_gram'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
        uni-gram -> identifies record as part of the uni-gram analysis
        bi-gram -> identifies record as part of the bi-gram analysis
        only_one_in_group -> record was not analysed because it was only one record
    m_dataframe['string_ones'] -> type: string; customer name, street numbers
        ones -> indicates records analyzed included tokens with a count of one
        non-ones -> indicates records analysed did not included tokesn with a count of one
    m_dataframe['string_tpi'] -> type: string; tpi number which is the customer idetnifier
    m_dataframe['string_country_code'] -> type: string; country code from the Cordys database
    m_dataframe['string_postal_code'] -> type: string; postal code from the Cordys database
    m_dataframe['string_city'] -> type: string; city from the Cordys database
    m_dataframe['string_address_to_dedup_00'] -> type: string; customer name, no street numbers
    m_dataframe['string_address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['string_address_to_dedup_02'] -> type: string; customer name, street numbers
    m_dataframe['string_address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                    numbers
    m_dataframe['list_tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
    m_dataframe['list_tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
    m_dataframe['list_tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
    m_dataframe['list_tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
        
    Important Info:
    None
    
    Return:
    object
    Type: pandas DataFrame
    Desc: this of the clusters that meet the evaluation criterea; same as passed as arguement
    m_dataframe['string_key'] -> type: string; customer name, no street numbers
    m_dataframe['string_gram'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['string_ones'] -> type: string; customer name, street numbers
    m_dataframe['string_tpi'] -> type: string; tpi number which is the customer idetnifier
    m_dataframe['string_country_code'] -> type: string; country code from the Cordys database
    m_dataframe['string_postal_code'] -> type: string; postal code from the Cordys database
    m_dataframe['string_city'] -> type: string; city from the Cordys database
    m_dataframe['string_address_to_dedup_00'] -> type: string; customer name, no street numbers
    m_dataframe['string_address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['string_address_to_dedup_02'] -> type: string; customer name, street numbers
    m_dataframe['string_address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                    numbers
    m_dataframe['string_tokens_00'] -> type: strings; uni-grams from address_to_dedup_00
    m_dataframe['string_tokens_01'] -> type: strings; uni-grams from address_to_dedup_01
    m_dataframe['string_tokens_02'] -> type: strings; uni-grams from address_to_dedup_02
    m_dataframe['string_tokens_03'] -> type: strings; uni-grams from address_to_dedup_03
    '''
    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # iterable declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_columns = ['string_key', 'string_gram', 'string_ones',  'string_tpi', 'string_country_code', 'string_postal_code', 
                            'string_city', 'string_address_to_dedup_00', 'string_address_to_dedup_01', 
                            'string_address_to_dedup_02', 'string_address_to_dedup_03', 'string_tokens_00', 
                            'string_tokens_01', 'string_tokens_02', 'string_tokens_03']

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # begin cleaning
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    for string_series in m_dataframe:
        if string_series[:-3] == 'list_tokens' or string_series[:-3] == 'list_bi_grams' or string_series == 'tuple_key':
            m_dataframe[string_series] = m_dataframe[string_series].apply(series_apply_iter_to_string)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return pandas.DataFrame(data = m_dataframe.values, columns = list_columns)

def format_df_dup_tpi(m_dataframe_tpi, m_list_df_columns, m_bool_build_addr, m_string_gram):
    '''
    this method formats the information in m_dataframe_tpi to fit into the destination table of duplicates
    
    Requirements:
    package pandas
    
    Inputs:
    m_dataframe_tpi
    Type: list
    Desc: the list of parameters for the clustering object
    if data is raw directly from the source table:
    m_dataframe_tpi['DB_NBR'] -> type: string; ??
    m_dataframe_tpi['CUSTOMER_NAME'] -> type: string; ??
    m_dataframe_tpi['SECONDARY_NAME'] -> type: string; ??
    m_dataframe_tpi['ADDRESS_1'] -> type: string; ??
    m_dataframe_tpi['ADDRESS_2'] -> type: string; ??
    m_dataframe_tpi['CITY'] -> type: string; ??
    m_dataframe_tpi['STATE'] -> type: string; ??
    m_dataframe_tpi['POSTAL_CODE'] -> type: string; ??
    m_dataframe_tpi['COUNTRY_CODE'] -> type: string; ??
    m_dataframe_tpi['COUNTRY'] -> type: string; ??
    m_dataframe_tpi['MAIL_ADDRESS_1'] -> type: string; ??
    m_dataframe_tpi['MAIL_ADDRESS_2'] -> type: string; ??
    m_dataframe_tpi['MAIL_CITY'] -> type: string; ??
    m_dataframe_tpi['MAIL_STATE'] -> type: string; ??
    m_dataframe_tpi['MAIL_POSTAL_CODE'] -> type: string; ??
    m_dataframe_tpi['PARETN_DB_NBR'] -> type: string; ??
    m_dataframe_tpi['SUPERSEDED_BY_DB_NBR'] -> type: string; ??
    m_dataframe_tpi['TYPE_CODE'] -> type: string; ??
    m_dataframe_tpi['CUSTOME_TYPE'] -> type: string; ??
    m_dataframe_tpi['GLOBAL_ACCOUNT_NAME'] -> type: string; ??
    m_dataframe_tpi['ACCOUNT_TYPE'] -> type: string; ??
    m_dataframe_tpi['ALLIANCE_PARTNER'] -> type: string; ??
    m_dataframe_tpi['INDUSTRY_CODE'] -> type: string; ??
    m_dataframe_tpi['INDUSTRY'] -> type: string; ??
    m_dataframe_tpi['INDUSTRY_GRP'] -> type: string; ??
    m_dataframe_tpi['MARKET'] -> type: string; ??
    m_dataframe_tpi['ACTIVE'] -> type: string; ??
    m_dataframe_tpi['VERIFIED'] -> type: string; ??
    m_dataframe_tpi['ACCOUNT_NUMBER'] -> type: string; ??
    m_dataframe_tpi['ACCOUNT_NAME'] -> type: string; ??
    m_dataframe_tpi['GEO_CODE'] -> type: string; ??
    m_dataframe_tpi['IS_PRIMARY'] -> type: string; ??
    m_dataframe_tpi['SOURCE'] -> type: string; ??
    m_dataframe_tpi['DIVISION'] -> type: string; ??
    m_dataframe_tpi['SE_ID'] -> type: string; ??
    m_dataframe_tpi['SE_CODE'] -> type: string; ??
    m_dataframe_tpi['SE_NAME'] -> type: string; ??
    m_dataframe_tpi['SE_ROLE'] -> type: string; ??
    m_dataframe_tpi['SE_FUNCTION'] -> type: string; ??
    m_dataframe_tpi['SLS_DISTRICT_CODE'] -> type: string; ??
    m_dataframe_tpi['SLS_DISTRICT'] -> type: string; ??
    m_dataframe_tpi['ETLInsertTS'] -> type: string; ??
    m_dataframe_tpi['ETLBatchID'] -> type: string; ??

    if the addresses are already built the dataframe is as the following:
    dataframes['tpi'] -> type: string; tpi number which is the customer identifier
    dataframes['country_code'] -> type: string; country code from the Cordys database
    dataframes['postal_code'] -> type: string; postal code from the Cordys database
    dataframes['city'] -> type: string; city from the Cordys database
    dataframes['address_to_dedup_00'] -> type: pandas Series; customer name, no street numbers
    dataframes['address_to_dedup_01'] -> type: pandas Series; customer name, secondary name, 
                                                                    no street numbers
    dataframes['address_to_dedup_02'] -> type: pandas Series; customer name, street numbers
    dataframes['address_to_dedup_03'] -> type: pandas Series; customer name, scondary name, street 
                                                                    numbers
    
    m_list_df_columns
    Type: list
    Desc: columns of the destination dataframe in the correct order to insert into the duplicates table
    m_list_df_columns[0] -> type: string; 'string_key'
    m_list_df_columns[1] -> type: string; 'string_gram'
    m_list_df_columns[2] -> type: string; 'string_ones'
    m_list_df_columns[3] -> type: string; 'string_tpi'
    m_list_df_columns[4] -> type: string; 'string_country_code'
    m_list_df_columns[5] -> type: string; 'string_postal_code'
    m_list_df_columns[6] -> type: string; 'string_city'
    m_list_df_columns[7] -> type: string; 'string_address_to_dedup_00'
    m_list_df_columns[8] -> type: string; 'string_address_to_dedup_01'
    m_list_df_columns[9] -> type: string; 'string_address_to_dedup_02'
    m_list_df_columns[10] -> type: string; 'string_address_to_dedup_03'
    m_list_df_columns[11] -> type: string; 'string_tokens_00'
    m_list_df_columns[12] -> type: string; 'string_tokens_01'
    m_list_df_columns[13] -> type: string; 'string_tokens_02'
    m_list_df_columns[14] -> type: string; 'string_tokens_03'

    m_bool_build_addr
    Type: boolean
    Desc: flag if the addresses need to be built

    m_string_gram
    Type: string
    Desc: the phrase to add to the gram columns
        
    Important Info:
    None
    
    Return:
    object
    Type: pandas DataFrame
    Desc: this of the clusters that meet the evaluation criterea
    m_list_df_columns[0] -> type: string; 'string_key'
    m_list_df_columns[1] -> type: string; 'string_gram'
    m_list_df_columns[2] -> type: string; 'string_ones'
    m_list_df_columns[3] -> type: string; 'string_tpi'
    m_list_df_columns[4] -> type: string; 'string_country_code'
    m_list_df_columns[5] -> type: string; 'string_postal_code'
    m_list_df_columns[6] -> type: string; 'string_city'
    m_list_df_columns[7] -> type: string; 'string_address_to_dedup_00'
    m_list_df_columns[8] -> type: string; 'string_address_to_dedup_01'
    m_list_df_columns[9] -> type: string; 'string_address_to_dedup_02'
    m_list_df_columns[10] -> type: string; 'string_address_to_dedup_03'
    m_list_df_columns[11] -> type: string; 'string_tokens_00'
    m_list_df_columns[12] -> type: string; 'string_tokens_01'
    m_list_df_columns[13] -> type: string; 'string_tokens_02'
    m_list_df_columns[14] -> type: string; 'string_tokens_03'
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_old_column_order = ['string_key', 'string_gram', 'string_ones', 'tpi', 'country_code', 'postal_code', 'city',
                                            'address_to_dedup_00', 'address_to_dedup_01', 'address_to_dedup_02',
                                            'address_to_dedup_03', 'tokens_00', 'tokens_01', 'tokens_02', 'tokens_03']

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # create columns addresses and/or tokens
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if m_bool_build_addr == True:
        '''
        after the build_address_to_dedup() method the dataframe is as below

        dataframes['tpi'] -> type: string; tpi number which is the customer identifier
        dataframes['country_code'] -> type: string; country code from the Cordys database
        dataframes['postal_code'] -> type: string; postal code from the Cordys database
        dataframes['city'] -> type: string; city from the Cordys database
        dataframes['address_to_dedup_00'] -> type: pandas Series; customer name, no street numbers
        dataframes['address_to_dedup_01'] -> type: pandas Series; customer name, secondary name, 
                                                                        no street numbers
        dataframes['address_to_dedup_02'] -> type: pandas Series; customer name, street numbers
        dataframes['address_to_dedup_03'] -> type: pandas Series; customer name, scondary name, street 
                                                                        numbers
        '''
        dataframe_temp_00 = build_addresses_to_dedup(m_dataframe_tpi)

        '''
        after the create_tokens() method the dataframe returned is as below

        dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
        dataframe['country_code'] -> type: string; country code from the Cordys database
        dataframe['postal_code'] -> type: string; postal code from the Cordys database
        dataframe['city'] -> type: string; city from the Cordys database
        dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
        dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                        no street numbers
        dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
        dataframe['address_to_dedup_03'] -> type: sgtring; customer name, scondary name, street 
                                                                        numbers
        dataframe['tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
        dataframe['tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
        dataframe['tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
        dataframe['tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
        '''
        dataframe_temp_00 = create_tokens(dataframe_temp_00, False)
    else:
        '''
        after the create_tokens() method the dataframe returned is as below

        dataframe['tpi'] -> type: string; tpi number which is the customer idetnifier
        dataframe['country_code'] -> type: string; country code from the Cordys database
        dataframe['postal_code'] -> type: string; postal code from the Cordys database
        dataframe['city'] -> type: string; city from the Cordys database
        dataframe['address_to_dedup_00'] -> type: string; customer name, no street numbers
        dataframe['address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                        no street numbers
        dataframe['address_to_dedup_02'] -> type: string; customer name, street numbers
        dataframe['address_to_dedup_03'] -> type: sgtring; customer name, scondary name, street 
                                                                        numbers
        dataframe['tokens_00'] -> type: list of strings; uni-grams from address_to_dedup_00
        dataframe['tokens_01'] -> type: list of strings; uni-grams from address_to_dedup_01
        dataframe['tokens_02'] -> type: list of strings; uni-grams from address_to_dedup_02
        dataframe['tokens_03'] -> type: list of strings; uni-grams from address_to_dedup_03
        '''
        dataframe_temp_00 = create_tokens(m_dataframe_tpi, False)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # create key, gram and ones columns
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    # create key
    dataframe_temp_00['string_key'] = dataframe_temp_00.apply(dataframe_apply_create_key, axis = 1)
    
    # create gram phrase
    dataframe_temp_00['string_gram'] = pandas.Series(data = [m_string_gram for x in \
                                                                range(0, len(dataframe_temp_00))])

    # create ones
    dataframe_temp_00['string_ones'] = pandas.Series(data = ['no-ones' for x in \
                                                                range(0, len(dataframe_temp_00))])

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # convert token series to string series
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    for string_series in dataframe_temp_00:
        if string_series[:-3] == 'tokens':
            dataframe_temp_00[string_series] = dataframe_temp_00[string_series].apply(
                                                                        series_apply_iter_to_string)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # order the dataframe
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    dataframe_temp_00 = dataframe_temp_00[list_old_column_order]

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return pandas.DataFrame(data = dataframe_temp_00.values, columns = m_list_df_columns)

def check_destination_tables(m_list_dest_conn, m_list_tables, m_list_df_columns):
    '''
    this method checks the destination table in if it doesn't exist create it
    
    Requirements:
    file SqlMethods
    package pandas
    
    Inputs:
    m_list_dest_conn
    Type: list
    Desc: the list of parameters to connect to the sql server
    list[0] -> type: string; user name
    list[1] -> type: string; sql sever name / host
    list[2] -> type: string; user password
    list[3] -> type: string; database to connect
    
    m_list_tables
    Type: list
    Desc: list of table names to check

    m_list_df_columns
    Type: list
    Desc: the list of dataframe series titles which will be the column names in the sql database
    m_list_df_columns[0] -> type: string; 'string_key'
    m_list_df_columns[1] -> type: string; 'string_gram'
    m_list_df_columns[2] -> type: string; 'string_ones'
    m_list_df_columns[3] -> type: string; 'string_tpi'
    m_list_df_columns[4] -> type: string; 'string_country_code'
    m_list_df_columns[5] -> type: string; 'string_postal_code'
    m_list_df_columns[6] -> type: string; 'string_city'
    m_list_df_columns[7] -> type: string; 'string_address_to_dedup_00'
    m_list_df_columns[8] -> type: string; 'string_address_to_dedup_01'
    m_list_df_columns[9] -> type: string; 'string_address_to_dedup_02'
    m_list_df_columns[10] -> type: string; 'string_address_to_dedup_03'
    m_list_df_columns[11] -> type: string; 'string_tokens_00'
    m_list_df_columns[12] -> type: string; 'string_tokens_01'
    m_list_df_columns[13] -> type: string; 'string_tokens_02'
    m_list_df_columns[14] -> type: string; 'string_tokens_03'
        
    Important Info:
    None
    
    Return:
    ??
    Type: ??
    Desc: ??
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_dest_column_names = ['date_analysis', 'string_active']
    list_dest_column_names.extend(m_list_df_columns)
    list_dest_column_dtypes = ['datetime', 'varchar(1)', 'varchar(100)', 'varchar(100)', 'varchar(50)', 'varchar(50)', 
                                'varchar(10)', 'varchar(100)', 'varchar(1000)', 'varchar(2000)', 'varchar(2000)', 'varchar(2000)', 
                                'varchar(2000)', 'varchar(max)', 'varchar(max)', 'varchar(max)', 'varchar(max)']
    list_create = list()
    if len(list_dest_column_names) == len(list_dest_column_dtypes):
        for int_index in range(0, len(list_dest_column_names)):
            list_create.append(list_dest_column_names[int_index] + ' ' + list_dest_column_dtypes[int_index])
    list_return = list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # db connections
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    sql_dest = SqlMethods(m_list_dest_conn)

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Check table and create 
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

    if sql_dest.bool_is_connected == True:
        for string_table in m_list_tables:
            bool_table_exists = sql_dest.table_exists(string_table)
            
            if bool_table_exists == False:
                list_create_dummy = sql_dest.create_table(string_table, list_create)
                list_return.append([list_create_dummy[0], string_table])      

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    sql_dest.close()
    del sql_dest

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_return

def update_active_flag(m_list_conn, m_list_tables):
    '''
    this method updates the active flag in the tables in list passed
    
    Requirements:
    package SqlMethods
    
    Inputs:
    m_list_conn
    Type: list
    Desc: connection parameters to connect to the sql server
    
    m_list_tables
    Type: list
    Desc: table names to update
        
    Important Info:
    None
    
    Return:
    object
    Type: list
    Desc: this of the clusters that meet the evaluation criterea
    list[x][0] -> type: boolean; flag if update occured without an error
    list[x][1] -> type: string; if boolean is True no error occured this will be an empty string; if Flase the string will
                        indicate the error that occured; the order in the list will reflect the order in m_list_tables
    '''
    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # iterables
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_errors = list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # db connections
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    sql_dest = SqlMethods(m_list_conn)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # update sql table
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if sql_dest.bool_is_connected == True:
        for string_sql_table in m_list_tables:    
            if sql_dest.table_exists(string_sql_table) == True:
                list_errors.append(sql_dest.update(string_sql_table, ['string_active'], ['0']))

        #------------------------------------------------------------------------------------------------------------------------------------------------------#
        # object cleanup
        #------------------------------------------------------------------------------------------------------------------------------------------------------#
    
        sql_dest.close()
        del sql_dest

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # return value
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    return list_errors

def send_results_to_dest(m_dataframe, m_list_sql_conn_args, m_string_table):
    '''
    this method takes the cluster results and and adds them to two table in the database
    
    Requirements:
    package pandas
    package SqlMethods
    
    Inputs:
    m_dataframe
    Type: pandas DataFrame
    Desc: the cluster results
    m_dataframe['string_key'] -> type: string; customer name, no street numbers
    m_dataframe['string_gram'] -> type: string; customer name, secondary name, 
                                                    no street numbers
        uni-gram -> identifies record as part of the uni-gram analysis
        bi-gram -> identifies record as part of the bi-gram analysis
        only_one_in_group -> record was not analysed because it was only one record
    m_dataframe['string_ones'] -> type: string; customer name, street numbers
        ones -> indicates records analyzed included tokens with a count of one
        non-ones -> indicates records analysed did not included tokesn with a count of one
    m_dataframe['string_tpi'] -> type: string; tpi number which is the customer idetnifier
    m_dataframe['string_country_code'] -> type: string; country code from the Cordys database
    m_dataframe['string_postal_code'] -> type: string; postal code from the Cordys database
    m_dataframe['string_city'] -> type: string; city from the Cordys database
    m_dataframe['string_address_to_dedup_00'] -> type: string; customer name, no street numbers
    m_dataframe['string_address_to_dedup_01'] -> type: string; customer name, secondary name, 
                                                                    no street numbers
    m_dataframe['string_address_to_dedup_02'] -> type: string; customer name, street numbers
    m_dataframe['string_address_to_dedup_03'] -> type: string; customer name, scondary name, street 
                                                                    numbers
    m_dataframe['string_tokens_00'] -> type: strings; uni-grams from address_to_dedup_00
    m_dataframe['string_tokens_01'] -> type: strings; uni-grams from address_to_dedup_01
    m_dataframe['string_tokens_02'] -> type: strings; uni-grams from address_to_dedup_02
    m_dataframe['string_tokens_03'] -> type: strings; uni-grams from address_to_dedup_03
    
    m_list_sql_conn_args
    Type: list
    Desc: list of sql connection arguments

    m_string_table
    Type: string
    Desc: table to insert data into

    Important Info:
    None
    
    Return:
    object
    Type: lists
    Desc: ??
    list[0] -> type: boolean; if the data was inserted successfully without an error
    list[1] -> type: string; if boolean is True then this string will be empty; if boolean is False this string will be
                    the error message from the insert
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # objects declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    sql_destination = SqlMethods(m_list_sql_conn_args)

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    datetime_now = datetime.now()
    string_datetime_now = datetime_now.strftime('%Y-%m-%d %H:%M:%S')

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # lists declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_dest_column_names = ['date_analysis', 'string_active']
    list_dest_column_names.extend(m_dataframe.columns.values)
    list_df_insert_dummy = list()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    string_active = '1'

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#     
    
    #--------------------------------------------------------------------#
    # create series of additional dimensions
    #--------------------------------------------------------------------#

    list_date_analysis = [string_datetime_now for x in range(0, len(m_dataframe))]
    series_date_analysis = pandas.Series(data = list_date_analysis, name = 'date_analysis')
    list_active = [string_active for x in range(0, len(m_dataframe))]
    series_active = pandas.Series(data = list_active, name = 'string_active')
    m_dataframe = m_dataframe.assign(date_analysis = series_date_analysis.values,
                                     index = m_dataframe.index)
    m_dataframe = m_dataframe.assign(string_active = series_active.values,
                                     index = m_dataframe.index)
    m_dataframe = m_dataframe[list_dest_column_names]

    #--------------------------------------------------------------------#
    # if connected insert the data
    #--------------------------------------------------------------------#

    if sql_destination.bool_is_connected == True:

        #--------------------------------------------------------------------#
        # insert into the database
        #--------------------------------------------------------------------#

        list_df_insert_dummy = sql_destination.insert(m_string_table, list_dest_column_names, 
                                                m_dataframe.values.tolist())

        #-------------------------------------------------------------#
        # variable / object cleanup
        #-------------------------------------------------------------#

        # sql server connection clean-up
        sql_destination.close()
        del sql_destination

    #-------------------------------------------------------------#
    # return value
    #-------------------------------------------------------------#

    return list_df_insert_dummy

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Main Method
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def main(list_args = []):
    '''
    this is the main method to conduct the cluster analysis to detect duplicates in the customer master
     
    Requirements:
    None
    
    Inputs:
    list_args
    Type: list
    Desc: arguements for login to the sql server
    list_args[0] -> type: string; user name
    list_args[1] -> type: string; password
      
    Important Info:
    None
    
    Return:
    None
    Type: None
    Description: None
    '''

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # object declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    # time declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    timer_main = Timer()

    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    # sequence declarations (list, set, tuple)
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    list_infsh_stage_conn = [list_args[0], r'gildv57\bi', list_args[1], r'InfoShareStage']
    #list_dest_conn = [list_args[0], r'gildv57\ML', list_args[1], r'Data_Science']
    list_dest_conn = [list_args[0], r'gildv57\ML', list_args[1], r'Data_Science']

    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    # variables declarations
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #string_cmdm_table = 'Cordys.Cmdm'
    string_cmdm_table = 'Cordys.TPI_STAGING'
    string_sql_table_dest_duplicates = 'CMDM.analysis_dup_uni_noones'
    string_sql_table_dest_dedupes = 'CMDM.analysis_dedup_uni_noones'

    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    # db connections
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    sql_infsh_stage = SqlMethods(list_infsh_stage_conn)

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # Start
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # get the data from Cordy's
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if sql_infsh_stage.bool_is_connected == True:
        print('getting customer data')
        timer_get_data = Timer()
        dataframe_dedup_tpi, dataframe_dup_tpi = get_customer_data(sql_infsh_stage, string_cmdm_table)
        sql_infsh_stage.close()
        timer_get_data.stop_timer('time to get customer data') 
        del timer_get_data, sql_infsh_stage

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # build dataframe with addresses to deduplicate
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    if 'dataframe_dedup_tpi' in locals():
        print('building addresses to deduplicate')
        timer_get_address = Timer()
        dataframe_dedup_addr, dataframe_dup_addr = build_addresses_to_dedup(dataframe_dedup_tpi)
        timer_get_address.stop_timer('time to build addresses')
        del timer_get_address

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # create tokens (uni-grams, bi-grams) from the dataframe addresses
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('creating tokens')
    timer_tokens = Timer()
    dataframe_to_cluster_no_ones = create_tokens(dataframe_dedup_addr, True)
    dataframe_to_cluster_ones = create_tokens(dataframe_dedup_addr, False)
    timer_tokens.stop_timer('time to create tokens')
    del timer_tokens

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # cluster the addresses to find the duplicates
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('beginning cluster analysis')
    timer_cluster = Timer()
    #list_cluster_results_ones_dup, list_cluster_results_ones_nondup = cluster_addresses(
    #                        dataframe_to_cluster_ones, True)
    list_cluster_results_no_ones_dup, list_cluster_results_no_ones_nondup = cluster_addresses(
                            dataframe_to_cluster_no_ones, False)
    timer_cluster.stop_timer('time to conduct the cluster analysis')
    del timer_cluster

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # place lists into two dataframes to insert into the server
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('beginning to orgazine results to add to sql server')
    timer_org = Timer()
    #dataframe_ones_dup = convert_to_dataframe(list_cluster_results_ones_dup)
    #dataframe_ones_nondup = convert_to_dataframe(list_cluster_results_ones_nondup)
    dataframe_no_ones_dup = convert_to_dataframe(list_cluster_results_no_ones_dup)
    dataframe_no_ones_nondup = convert_to_dataframe(list_cluster_results_no_ones_nondup)
    timer_org.stop_timer('time to organize cluster results')
    del timer_org

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # cleaning clustered dataframes
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('begin cleaning dataframes')
    timer_clean_df = Timer()
    dataframe_dup = clean_dataframe(dataframe_no_ones_dup)
    dataframe_nondup = clean_dataframe(dataframe_no_ones_nondup)
    timer_clean_df.stop_timer('time to clean duplicate and non-duplicate dataframes')
    del timer_clean_df

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # format dataframes duplicates by TPI and addresses to insert into the duplicates table
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('formatting the tpi and address duplicates to insert into the duplicates tables')
    timer_format_tpi_addr = Timer()
    if len(dataframe_dup_tpi) > 0:
        dataframe_dup_tpi = format_df_dup_tpi(dataframe_dup_tpi, dataframe_dup.columns.values, True, 
                                              'tpi_dup')
    else:
        dataframe_dup_tpi = pandas.DataFrame()

    if len(dataframe_dup_addr) > 0:
        dataframe_dup_address = format_df_dup_tpi(dataframe_dup_addr, dataframe_dup.columns.values,
                                                        False, 'addr_dup')
    else:
        dataframe_dup_address = pandas.DataFrame()
    timer_format_tpi_addr.stop_timer('time to format tpi and address duplicates')
    del timer_format_tpi_addr

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # merge dataframes
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('begining concatinating dataframes of duplicates')
    timer_concat = Timer()
    dataframe_dup = pandas.concat([dataframe_dup, dataframe_dup_tpi, dataframe_dup_address])
    del dataframe_dup_tpi, dataframe_dup_address
    timer_concat.stop_timer('time to concat duplicates')
    del timer_concat

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # test code; send to CSV to examine dataframes
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    #print('copying dataframes to csv files for manual inspection')
    #timer_csv = Timer()
    #string_path = os.path.abspath('./') + os.path.sep
    #dataframe_dup.to_csv(string_path + 'df_dup_unig_noones.csv')
    #dataframe_nondup.to_csv(string_path + 'df_nondup_unig_noones.csv')
    #timer_csv.stop_timer('time to send dataframes to csv files')
    #del timer_csv

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # check destination tables
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('checking destination tables')
    timer_dest_tables = Timer()
    list_dest_tables = [string_sql_table_dest_duplicates, string_sql_table_dest_dedupes]
    check_destination_tables(list_dest_conn, list_dest_tables, dataframe_dup.columns.values)
    timer_dest_tables.stop_timer('time to check or create destination tables')
    del timer_dest_tables

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # replace null-values with empty string
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    print('replacing null values before insert into database')
    timer_replace_null = Timer()
    array_bool_null = dataframe_dup.isnull()
    dataframe_dup[array_bool_null] = ''
    del array_bool_null

    array_bool_null = dataframe_nondup.isnull()
    dataframe_nondup[array_bool_null] = ''
    del array_bool_null
    timer_replace_null.stop_timer('time to replace null values in dataframes')
    del timer_replace_null

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # send results to sql database for analysis
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    timer_sql_insert = Timer()
    # update tables active flag
    print('set active flag to zero')
    list_update_dummy = update_active_flag(list_dest_conn, [string_sql_table_dest_dedupes,
                                                            string_sql_table_dest_duplicates])
    
    # insert results into sql table
    print('inserting cluster results to database')
    list_insert_results_dup = send_results_to_dest(dataframe_dup, list_dest_conn, 
                                                   string_sql_table_dest_duplicates)
    list_insert_results_nondup = send_results_to_dest(dataframe_nondup, list_dest_conn, 
                                                      string_sql_table_dest_dedupes)
    timer_sql_insert.stop_timer('time to insert analysis results into database')
    del timer_sql_insert

    # print insert results
    if list_insert_results_dup[0] == True:
        print('duplicates inserted successfully')
    else:
        print('duplicates has insert errors: %s', list_insert_results_dup[1])

    if list_insert_results_nondup[0] == True:
        print('non-duplicates inserted successfully')
    else:
        print('non-duplicates has insert errors: %s', list_insert_results_nondup[1])

    timer_main.stop_timer('stop of main method')
    del timer_main

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #
    # sectional comment
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # variable / object cleanup
    #------------------------------------------------------------------------------------------------------------------------------------------------------#

    pass

# call Main
#if __name__ == '__main__':
#    main()