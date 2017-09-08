#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# File / Package Import
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#    

import tkn_transform

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Global Methods
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#    



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Classes
#
# PuncFulter()
# Desc: this is a puncatiaon filter which is a subclass of TransformBase(); this class removes all the tokens
#                which are ONLY punucations as defined in the class
#
# AbbrvFilter()
# Desc: this class is an abbreviation filter which is a subclass of TransformBase(); this class revmoves tokens
#                that are in the abbreviation list in the class
# 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#    

class PuncFilter(tkn_transform.TransformBase):
    ###############################################################################################
    ###############################################################################################
    # 
    # this class implements a punctuation filter for the tokens in natural language processing.  This class is a subclass
    # of TransformBase oringaly developed by Wes Soloman from Saxoney Partners.  
    #
    # Requirements:
    # file tkn_transform
    # 
    # Important Info:
    # as part of a base class the TransformFiltBase the implemenation assumes the functions calls are with both a
    # token and flags set().  In methods _reject_tkn() and run() need to be called with both a tokens and tags set for
    # the superclass and global methods to function properly.
    #
    # methods:
    # _reject_tkn()
    # inputs:
    # tkn -> type: set; tokens to check for punction
    # tag -> type: set; tags to account for if present, not used
    # return: boolean
    #        True: if all the characters in the token are in the set punctuation
    #        False: if at least one character in the token is not in the set punctuation
    # desc: check all the characters in a token are a punctaion of some type (in set punctuation)
    #
    # run()
    # inputs:
    # _reject_tkn -> type: method; checking for a token if all the characters are punctuation
    # tkns -> type: set; tokens
    # tags -> type: set; flags for a token (token with two meanings therefore two tokens are needed)
    # return: list_tokens, list_tags
    #        list_tokens -> type: list; transformed tokens
    #        list_tags -> type: list; tags for each token
    # desc: calls the global method filt_run() which will call the method _reject_tkn() in this class; which returns a token
    #            and tags list; the tokens list will be the tokens in which all the characters in that token are NOT punctuations
    #            as defined by the set punctuation in this class
    #            example: 
    #                ['great!', '57%9@z', '--', '!', '\\', ','] filters to ['great!', '57%9@z']
    ###############################################################################################
    ###############################################################################################    

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # initialize / constructor
    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    def __init__(self, flgset = set()):
        # super initializer / constructer from TransformerBase
        super(PuncFilter, self).__init__(flgset, tkn_transform.replace_run)

        # set of punctuation characters to check for in token
        self.punctuation = {'`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', '{', '[', '}', ']', '\\', '|', ':', ';', 
                                        '"', "'", '<', ',', '>', '.', '?', '/'}

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # method to evaluate token rejection based on punctuation
    #------------------------------------------------------------------------------------------------------------------------------------------------------#        
    def _reject_tkn(self, tkn, tag=None):
        # split token from flag / tag
        rtkn, flg = self.flg_splitr.split(tkn)
        punctuation_bools = set()
        
        # create boolean set for each character in token
        for char in rtkn:
            punctuation_bools.add(char in self.punctuation)

        # test if every character in the token is a punctuation
        if punctuation_bools == {True}:
            return True
        return False            

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # method to execute in global method run_pipeline or indivudual execution of class
    # filt_run() is a global method
    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    def run(self, tkns, tags=None):
        return tkn_transform.filt_run(self._reject_tkn, tkns, tags)

class AbbrvFilter(tkn_transform.TransformBase):
    ###############################################################################################
    ###############################################################################################
    # 
    # this class implements a abbrevation filter for the tokens in natural language processing.  This class is a subclass
    # of TransformBase oringaly developed by Wes Soloman from Saxoney Partners.  
    #
    # Requirements:
    # file tkn_transform
    # 
    # Important Info:
    # as part of a base class the TransformBase the implemenation assumes the functions calls are with both a
    # token and flags set().  In methods _reject_abbrv() and run() need to be called with both a tokens and tags set for
    # the superclass and global methods to function properly.
    #
    # methods:
    # _reject_abbrv()
    # inputs:
    # tkn -> type: set; tokens to check for punction
    # tag -> type: set; tags to account for if present, not used
    # return: boolean
    #        True: if token is in the abbreviation set
    #        False: if token is not in the abbreviation set
    # desc: check the tokens to determine if there is an abbreviation and reject it
    #
    # run()
    # inputs:
    # _reject_abbrv -> type: method; checking for a token if all the characters are punctuation
    # tkns -> type: set; tokens
    # tags -> type: set; flags for a token (token with two meanings therefore two tokens are needed)
    # return: list_tokens, list_tags
    #        list_tokens -> type: list; transformed tokens
    #        list_tags -> type: list; tags for each token
    # desc: calls the global method filt_run() which will call the method _reject_abbrv() in this class; which returns a token
    #            and tags list; the tokens list will be the tokens in which all the characters in that token are NOT abbreviations
    #            as defined by the set abbreviation in this class
    #            example: 
    #                ["she's", "don't", "we've"] filters to ["she", "do", "we"]
    ###############################################################################################
    ###############################################################################################    

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # initialize / constructor
    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    def __init__(self, flgset = set()):
        # super initializer / constructer from TransformerBase
        super(AbbrvFilter, self).__init__(flgset, tkn_transform.replace_run)

        # set of punctuation characters to check for in token
        self.abbrevations = {"'s", "'ve", "n't"}

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # method to evaluate token rejection based on punctuation
    #------------------------------------------------------------------------------------------------------------------------------------------------------#        
    def _reject_abbrv(self, tkn, tag=None):
        # split token from flag / tag
        rtkn, flg = self.flg_splitr.split(tkn)

        # test if token is the abbreviation set
        if rtkn in self.abbrevations:
            return True
        return False        

    #------------------------------------------------------------------------------------------------------------------------------------------------------#
    # method to execute in global method run_pipeline or indivudual execution of class
    # filt_run() is a global method
    #------------------------------------------------------------------------------------------------------------------------------------------------------#    
    def run(self, tkns, tags=None):
        return tkn_transform.filt_run(self._reject_abbrv, tkns, tags)