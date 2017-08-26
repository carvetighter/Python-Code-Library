###############################################################################################
###############################################################################################
#
# This import file <DataPrep.py> contains methods to retrieve, clean, slice, sort and convert data in preperation for
# clusttering, classifying, evaluating and visualizing
#
# Requirements:
# package pandas
# package numpy
# package time
# package sklearn.feature_extraction.text
# file SqlMethods.py
#
# Methods included:
# GetData()
#    - retrieves data from a sql database and returns a dataframe
#
# CleanData()
#    - specific method to clean data for the CPBB model and returns a dataframe
#
# SliceData()
#    - specific method to slice data based on a list passed
#
# ConvertToTFIDF()
#    - trade a TFIDF matrix based on a dataframe passed, this is for a specfic column
#    - retunrs a list of different dense matrix, sparse matrix, dataframe
#
# SortDict()
#    - sorts a dictionary and returns a list, required for ConvertToTFIDF()
#
# NoSpaces()
#    - removes the spaces on the left and right of the string and any spaces that are more than one
#    - within the string
#
# Important Info:
# package pymssql will be imported with the file SqlMethods.py
###############################################################################################
###############################################################################################

# package import
from SqlMethods import SqlGenSelectStatement
import pandas, numpy, time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def GetData(m_sql_connection, m_select, m_str_from, m_str_where):
    ###############################################################################################
    ###############################################################################################
    #
    # this method gets the data from an instance of SQL Server on server GILDV319
    #
    # Requirements:
    # package pandas
    # package pymssql
    # package time
    #
    # Inputs:
    # m_sql_connection
    # Type: pymssql connection object
    # Desc: this is the connection to the sql server
    #  
    # m_sql_query
    # Type: string
    # Desc: sql query statement
    #  
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: pandas.DataFrame object
    # Desc: the raw data from the test set of data
    ###############################################################################################
    ###############################################################################################

    # connection info to the database
    string_query = SqlGenSelectStatement(m_str_select = m_select, m_str_from = m_str_from, m_str_where = m_str_where)

    # query the database
    time_qry_start = time.perf_counter()
    dataframeRaw = pandas.DataFrame(pandas.read_sql(string_query, m_sql_connection))
    time_qry = time.perf_counter() - time_qry_start

    # print query information
    print('Record count:', dataframeRaw['str_OrderNumber'].count())
    print('Query time (HH:MM:SS):', time.strftime('%H:%M:%S', time.gmtime(time_qry)))

    # return data frame
    return dataframeRaw

def CleanData(dataframeData):
    ################################################################################################
    ###############################################################################################
    #
    # this method cleans the data in the dataframe passed; looking in the columns identified to [str_ItemDescription] and the 
    # [str_NounCodeDescription] with spaces and numpy.nan types; this is broken up into two sub-methods to look at the 
    # numpy.nan potential values and the spaces
    #
    # Requirements:
    # package numpy
    # package pandas
    # method NoSpaces()
    #
    # Inputs:
    # dataframeData
    # type: pandas.DataFrame
    # desc: the dataframe which is the raw data from the database
    #  
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: pandas.DataFrame object
    # Desc: the clean data based on the cleaning criterea
    ###############################################################################################
    ###############################################################################################
    # start time
    time_clean_start = time.perf_counter()

    # lists
    list_columns_comma = ['str_ItemDescription']
    list_nan_space = [numpy.nan, 'nan', ',', ';']
    list_phrases_delete = ['Unassigned', 'Miscellaneous', 'Other', 'Others', 'MISCELLANEOUS', 'UNASSIGNED']
    list_date_columns = ['date_CreationDate']
    list_string_data = list()
    
    # convert data from dataframe to array
    list_data = [tuple(x) for x in dataframeData.values]

    # replace spaces inside each element of data
    # replaces multiple spaces with only one space
    for i in range(0, len(list_data)):
        list_string_data.append(tuple(NoSpaces(str(j)) for j in list_data[i]))

    # create new data frame
    dataframe_new = pandas.DataFrame(data = list_string_data, columns = dataframeData.columns)

    # replace commas with spaces, replace empty cells with a space, replace numpy.nan with space
    for col in list_columns_comma:
        # phrases to replace with a space
        for k in list_nan_space:
            dataframe_new[col] = dataframe_new[col].str.replace(str(k), ' ')

        # phrases to delete, replace with empty string, ''
        for k in list_phrases_delete:
            dataframe_new[col] = dataframe_new[col].str.replace(k, '')

        # take out the spaces at the ends of the segement
        dataframe_new[col] = dataframe_new[col].str.strip()

     # ensure that the [str_ItemDescription] has something in it and is not null
    dataframe_new = dataframe_new[dataframe_new.str_ItemDescription.notnull()]

    # convert the dates to a date data type
    for col in list_date_columns:
        dataframe_new[col] = pandas.to_datetime(dataframe_new[col])

    # stop time
    time_clean_duration = time.perf_counter() - time_clean_start

    # print time information
    print('Record count:', dataframe_new['str_OrderNumber'].count())
    print('Clean time (HH:MM:SS):', time.strftime('%H:%M:%S', time.gmtime(time_clean_duration)))

    # the dataframe returned, clean data
    return dataframe_new

def SliceData(dataframeData, list_slice_criterea):
    ################################################################################################
    ###############################################################################################
    #
    # the method will slice the data to be seperate the 'Miscellaneous', 'Uassigned', 'Others'
    #
    # Requirements:
    # package pandas
    #
    # Inputs:
    # dataframeData
    # type: pandas.DataFrame
    # desc: the dataframe which is the clean data from the database
    #  
    # list_slice_criterea
    # type: list object
    # desc: the list of phrases or criterea to create a new dataframe
    #  
    # Important Info:
    # a new column is created in the data frame [str_ClassColumn01] which is the combination of the columns
    # [str_ItemDescription]
    #
    # Return:
    # object
    # Type: pandas.DataFrame object
    # Desc: the records which have the the unassigned values
    ###############################################################################################
    ###############################################################################################

    # start time
    time_slice_start = time.perf_counter()

    # for intellisense in visual studio
    dataframeData = pandas.DataFrame(dataframeData)

    # get the data frame to run the cluster algorithm on
    dataframe_cluster = pandas.DataFrame(dataframeData.loc[dataframeData['str_ItemCategory'].isin(list_slice_criterea)])

    # create new column to cluster on which combines columns [str_ItemDescription] & [str_NounCodeDescription]
    dataframe_cluster['str_ClassColumn01'] = dataframe_cluster['str_ItemDescription']

    # strip spaces off the end
    dataframe_cluster['str_ClassColumn01']  = dataframe_cluster['str_ClassColumn01'] .str.strip()

    # ensure [str_ClassColumn01] is not null
    dataframe_cluster = dataframe_cluster[dataframe_cluster.str_ClassColumn01.notnull()]

    # duration time
    time_slice_duration = time.perf_counter() - time_slice_start
    print('Record count:', dataframe_cluster['str_OrderNumber'].count())
    print('Slice time (HH:MM:SS):', time.strftime('%H:%M:%S', time.gmtime(time_slice_duration)))

    # return dataframe
    return dataframe_cluster

def ConvertToTFIDF(dataframeData):
    ################################################################################################
    ###############################################################################################
    #
    # the method will take the column [str_ClassColumn01] from the dataframe and create a TFIDF matrix to cluster on
    #
    # Requirements:
    # package sklearn.feature_extranction.text
    # package pandas
    # method SortDict()
    #
    # Inputs:
    # dataframeData
    # type: pandas.DataFrame
    # desc: the dataframe which is the clean data
    #  
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: list
    # Desc: the list contains three objects:
    # list[0] -> sparse matrix of the TFIDF matrix
    # list[1] -> dense matrix of the TFIDF matrix
    # list[2] -> dataframe of the TFIDF matrix
    ###############################################################################################
    ###############################################################################################

    # start time
    time_tfidf_start = time.perf_counter()

    # for visual studio for method and type checking
    dataframeData = pandas.DataFrame(dataframeData)

    # lists
    list_return = list()
    
    # create the count vector matix from the works in [str_ClassColumn01]
    cnt_vect = CountVectorizer(min_df = 1)
    class_col_cnt_vect = cnt_vect.fit_transform(dataframeData['str_ClassColumn01'])
    
    # create the TFIDF matrix
    tfidf_trans = TfidfTransformer(smooth_idf = False)
    tfidf_sparse = tfidf_trans.fit_transform(class_col_cnt_vect)

    # create dense TFIDF matrix
    tfidf_dense = tfidf_sparse.todense()

    # get columns for the dataframe
    list_sorted_v = SortDict(cnt_vect.vocabulary_ , 1)
    list_columns_v = [i[0] for i in list_sorted_v]

    # create data frame
    dataframe_tfidf = pandas.DataFrame(data = tfidf_dense, columns = list_columns_v)
    
    # create return list
    list_return.append(tfidf_sparse)
    list_return.append(tfidf_dense)
    list_return.append(dataframe_tfidf)

    # time duration
    time_tfidf_duration = time.perf_counter() - time_tfidf_start
    print('TFIDF time (HH:MM:SS):', time.strftime('%H:%M:%S', time.gmtime(time_tfidf_duration)))

    # return list
    return list_return

def SortDict(dict_org, int_index = 0, bool_descending = False):
    ###############################################################################################
    ###############################################################################################
    #
    # this method sorts a dictionary ascending or descending and returns a list of tuples
    #
    # Requirements:
    # None
    #
    # Inputs:
    # dict_org
    # Type: dictionary
    # Desc: the dictionary to be sorted
    #
    # int_index
    # Type: integer
    # Desc: the index of the dictionary to sort; 0 will sort by the keys, 1 will sort by the values
    # 
    # bool_return_list
    # Type: boolean
    # Desc: flag flag to return list or dictionary, default is dictionary
    #    
    # Important Info:
    # None
    #
    # Return:
    # object
    # Type: list
    # Desc: list is returned a list of tuples is returned
    ###############################################################################################
    ###############################################################################################
    
    # create a sorted list from the dictionary
    list_sorted = sorted(dict_org.items(), key = lambda x: x[int_index], reverse = bool_descending)
       
    return list_sorted

def NoSpaces(string_org):
    ###############################################################################################
    ###############################################################################################
    #
    # this subroutine takes a string and removes the spaces on the outside of the string (left and right) and any spaces
    # in the string that are more than one; returns the new string
    #
    # Requirements:
    # None
    #
    # Inputs:
    # string_org
    # Type: string
    # Desc: the string to take out the spaces
    #
    # Important Info:
    # None
    #
    # Return:
    # variable
    # Type: string
    # Desc: the new string with no spaces inside and outside
    ###############################################################################################
    ###############################################################################################

    # take out leading and trailing spaces
    string_org = string_org.strip()

    # split the string on a space
    list_new = string_org.split(' ')

    # return a new string
    # if the item in the list is more than length zero join with a space
    return ''.join(item for item in list_new if item)