#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This is an auxiliary code for tensor-tensor product (t-product) operations between third-order tensors
    based on the t-product algebra depicted in https://github.com/canyilu/Tensor-tensor-product-toolbox and
    in the paper published by Pena-Pena et al. entitled 't-HGSP: Hypergraph Signal Processing Using t-Product 
    Tensor Decompositions' DOI: 10.1109/TSIPN.2023.3276687.
"""

__author__ = "Breno Bispo"
__contact__ = "breno.bispo@ufpe.br"
__date__ = "2024/09/12"
__status__ = "Production"


####################
# Review History   #
####################

# Reviewed and Updated by Breno Bispo 20201103

####################
# Libraries        #
####################

# Third party imports
import numpy as np
from scipy.fft import fft, ifft
import numpy.linalg as lg

class ttensor:
    """ Declaration of a class called 'ttensor', whose inputs can be the field 'data' or 'fft'.
    If a class 'ttensor' is initialized using the field 'data' as input, then the discrete Fourier
    transform of its tubal scalars is automatticaly computed and stored in the field 'fft'. Otherwise, 
    if a class 'ttensor' is initialized using the field 'fft' as input, then the inverse Fourier transorm 
    of its tubal scalars is automaticaly computed and stored in the field 'data'.

    Parameters
    ----------
    data : 3-order numpy array 

    or

    fft: 3-order numpy array

    """
    
    def __init__(self, tol=1e-6, **kwargs):
       
        if 'data' not in kwargs and 'fft' not in kwargs:
            raise ValueError("Declare 'data' or 'fft' fields!")
        
        if 'fft' not in kwargs:
            f = fft(kwargs['data'], axis=2)
        else:
            f = kwargs['fft']
        
        if 'data' not in kwargs:
            d = ifft(kwargs['fft'], axis=2)
        else:
            d = kwargs['data']

        if not np.isrealobj(d): 
            if np.sum(np.abs(np.imag(d))) < tol:
                self.data = np.real(d)
            else:
                self.data = d
        else:
            self.data = d 
        
        if not np.isrealobj(f): 
            if np.sum(np.abs(np.imag(f))) < tol:
                self.fft = np.real(f)
            else:
                self.fft = f
        else:
            self.fft = f  


def round2real(data, tol=1e-6):

    """Verify if the tensor 'data' is real by summing the absolute values of the imaginary parts of all entries.
    If the sum is less than 'tol', then this function returns only the real parts of the entries.
    Otherwise, returns the original input.
    
    Parameters
    ----------
    data: numpy array

    tol: float

    """

    if np.isrealobj(data):
        return data
    
    if np.sum(np.abs(np.imag(data))) < tol:
        data_real = np.real(data)
        return data_real
    else:
        return data

def tensor2sym(A):

    """Return the symmetrized tensor of input 'A'

    Parameters
    ----------
    A: 3-order numpy array
    
    """

    n = A.shape
    As = np.zeros((n[0],n[1],1))
    As = 0.5*np.concatenate((As, A, np.flip(A,axis=2)), axis=2)
    return As

def ttran(A):
    """Return the conjugate tensor transpose
    
    Parameters
    ----------
    A: 3-order numpy array

    """
    n = A.data.shape
    At = A.data.copy()
    if np.isrealobj(At):
        At[:,:,0] = A.data[:,:,0].T
        for k in range(1,n[2]):
            At[:,:,k] = A.data[:,:,n[2]-k].T
    else:
        At[:,:,0] = A.data[:,:,0].conj().T
        for k in range(1,n[2]):
            At[:,:,k] = A.data[:,:,n[2]-k].conj().T
            
    B = ttensor(data=At)
    return B

def tinv(A):
    """Return the tensor inverse

    Parameters
    ----------
    A: 3-order numpy array
    """

    n = A.data.shape
    C_fft = A.fft.copy()
    for k in range(n[2]):
        C_fft[:,:,k] = lg.inv(A.fft[:,:,k])
    
    C = ttensor(fft=C_fft)
    return C

def tprod(*args):
    """Return a class 'ttensor' consisting of the tensor-tensor product between arguments.

    Parameters
    ----------
    args: 3-order numpy arrays
    """
    
    args_len = len(args)

    for idx in range(args_len):
        n = args[idx].fft.shape
        m = args[idx+1].fft.shape
        if n[1] != m[0] or n[2] != m[2]:
            raise ValueError("Inner tensors dimensions must agree.")
        if idx == len(args)-2:
            break

    C_fft = args[0].fft.copy()
    for idx in range(args_len):    
        for k in range(n[2]):
            C_fft[:,:,k] = C_fft[:,:,k] @ args[idx+1].fft[:,:,k]
        if idx == args_len-2:
            break
    
    C = ttensor(fft=C_fft)
    return C

def teig(A, increasing=True):

    """Return the t-eigendecomposition of 'A', where 'U' is a set of eigen-matrices, 'Lambda_ttensor' is a f-diagonal tensor
    consisting of the eigen-tuples, and 'Lambda_norm_ttensor' is a f-diagonal tensor consisting of the normalized eigen-tuples.

    Parameters
    ----------
    A: 3-order numpy array
    
    increasing: boolean
        If True, the eigen-tuples are ordered increasingly. Otherwise, they are ordered decreasingly.
    """

    u = A.fft.copy()
    L = A.fft.copy()
    Ln = A.fft.copy()
    n = u.shape
    Lambda = np.zeros((n[0],n[2]))
    Lambda_norm = np.zeros((n[0],n[2]))
    for k in range(n[2]):
        eigvalues, eigvectors = lg.eigh(A.fft[:,:,k], UPLO='U')
        Lambda[:,k] = round2real(eigvalues)
        u[:,:,k] = round2real(eigvectors)
        Lambda_norm[:,k] = Lambda[:,k]/np.abs(np.max(Lambda[:,k]))
        idx = Lambda_norm[:,k].argsort()
        if increasing:
            idx = np.flip(idx)
        Lambda[:,k] = Lambda[idx,k]
        Lambda_norm[:,k] = Lambda_norm[idx,k]
        L[:,:,k] = np.diag(Lambda[:,k])
        Ln[:,:,k] = np.diag(Lambda_norm[:,k])
        u[:,:,k] = u[:,idx,k]
    

    U = ttensor(fft=u)
    Lambda_ttensor = ttensor(fft=L)
    Lambda_norm_ttensor = ttensor(fft=Ln)
    return U, Lambda_ttensor, Lambda_norm_ttensor

def teye(n, n3):
    """Return an 3-order identity tensor with 'n' by 'n' by 'n3'

    Parameters
    ----------
    n: integer
    
    n3: integer
    """
    I = np.zeros((n, n, n3))
    I[:,:,0] = np.eye(n)
    return I

def degree_tensor(A):
    """Return the degree tensor of 'A'

    Parameters
    ----------
    A: 3-order numpy array

    """
    n = A.shape
    D = np.zeros((n))
    s = np.sum(A,axis=2)
    s = np.sum(s,axis=1)
    for i in range(n[0]):
        D[i,i,i] = s[i]
    return D

def laplacian_tensor(A):
    """Return the laplacian tensor of 'A'

    Parameters
    ----------
    A: 3-order numpy array
    
    """
    return degree_tensor(A) - A