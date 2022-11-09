import numpy as np
import pandas as pd
import re
import sparse_dot_topn.sparse_dot_topn as ct

from ftfy import fix_text
from scipy.sparse import csr_matrix
from unidecode import unidecode

def clean_city(city):
    ''' Nettoie les noms de villes'''
    city= str(city)
    chars_to_remove = [")","(",".","|","[","]","{","}","'","-",",","1","2","3","4","5","6","7","8","9","0"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    city = re.sub(rx, '', city) 
    city = city.upper()
    city = city.replace("CEDEX","")
    city = city.replace("SAINT ","ST ")
    city = city.replace("SUR ","S ")
    city = city.strip()
    city = re.sub(' +',' ',city).strip()

    return city

def clean_company(string):
    '''nettoie les noms de companies'''
    string = str(string)
    string = string.lower()
    string = fix_text(string) # corrige les erreurs d'encodage voir si doublon avec unidecode
    string = unidecode(string)
    chars_to_remove = [")","(",".","|","[","]","{","}","'","-",","]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']' # supprime la ponctuation
    string = re.sub(rx, '', string) 
    string = string.replace('&', 'et')
    forme_juridique =["sa", "sarl", "ltd","limited", "company", "incorporated", "corporation", "scp",
                      "spa", 'corp', "inc","gmbh","eurl","llp","lllp","snc","sas","eg","societe", "srl","association","groupe"

                  ]
    clean_string = ""
    for token in string.split(" "):
        if token not in forme_juridique:
            clean_string =clean_string+ token+" "
    string = re.sub(' +',' ',clean_string).strip() # get rid of multiple spaces and replace with a single 
    
    return string

def ngrams(string, n=3):
    '''converti en ngrams'''
    string = ' '+ string  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]    

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})