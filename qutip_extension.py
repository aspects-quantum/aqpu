import numpy as np

def adjoint_superoperator(H):
    '''
    Returns the Commutator with H in the form of a superoperator
    '''
    dim = H.shape[0]
    return np.kron(H,np.eye(dim))-np.kron(np.eye(dim),np.transpose(H))

def anticommutator_superoperator(Pi):
    '''
    Returns the Anticommutator with Pi in superoperator form
    '''
    dim = Pi.shape[0]
    return np.kron(Pi,np.eye(dim))+np.kron(np.eye(dim),Pi.T)

def diss(J):
    '''
    Returns the superoperator of the dissipator with J as the collapse operator.
    '''
    dim = J.shape[0]
    return np.kron(J,np.conj(J))-0.5*(np.kron(np.conj(J.T).dot(J),np.eye(dim))+np.kron(np.eye(dim),np.conj(J.T).dot(J)))