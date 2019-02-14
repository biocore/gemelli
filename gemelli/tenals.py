import numpy as np
from numpy.random import randn
from numpy.linalg import norm

def tenals(TE, E, r = 3, ninit = 50, nitr = 50, tol = 1e-8):
    
    #start
    n1,n2,n3 = TE.shape
    p = np.count_nonzero(TE)

    normTE = 0 
    for i3 in range(n3):
        normTE = normTE + norm(TE[:,:,i3])**2

    # initialization by Robust Tensor Power Method (modified for non-symmetric tensors)
    U01 = np.zeros((n1,r))
    U02 = np.zeros((n2,r))
    U03 = np.zeros((n3,r))
    S0 = np.zeros((r,1))
    for i in range(r):
        tU1 = np.zeros((n1,ninit))
        tU2 = np.zeros((n2,ninit))
        tU3 = np.zeros((n3,ninit))
        tS = np.zeros((ninit,1))
        for init in range(ninit):
            [tU1[:,init], tU2[:,init], tU3[:,init]] = RTPM(TE-CPcomp(S0,U01,U02,U03), nitr)  
            tU1[:,init] = tU1[:,init]/norm(tU1[:,init])
            tU2[:,init] = tU2[:,init]/norm(tU2[:,init])
            tU3[:,init] = tU3[:,init]/norm(tU3[:,init])
            tS[init] = TenProj(TE-CPcomp(S0,U01,U02,U03),tU1[:,[init]],tU2[:,[init]],tU3[:,[init]])
        [C, I] = np.max(tS,axis=0)[0], np.argmax(tS, axis=0)[0]
        U01[:,i] = tU1[:,I]/norm(tU1[:,I])
        U02[:,i] = tU2[:,I]/norm(tU2[:,I])
        U03[:,i] = tU3[:,I]/norm(tU3[:,I])
        S0[i] = TenProj(TE-CPcomp(S0,U01,U02,U03),U01[:,[i]],U02[:,[i]],U03[:,[i]])

    # apply alternating least squares        
    V1 = U01.copy()
    V2 = U02.copy()
    V3 = U03.copy()
    S = S0.copy()
    for itrs in range(nitr):
        for q in range(r):
            S_ = S.copy()
            S_[q] = 0 
            A = np.multiply(CPcomp(S_,V1,V2,V3),E)
            v1 = V1[:,q].copy()
            v2 = V2[:,q].copy()
            v3 = V3[:,q].copy()
            V1[:,q] = 0
            V2[:,q] = 0
            V3[:,q] = 0
            den1 = np.zeros((n1,1))
            den2 = np.zeros((n2,1))
            s = S[q]
            for i3 in range(n3):
                V1[:,q] = V1[:,q] + np.multiply(v3[i3],np.matmul((TE[:,:,i3]-A[:,:,i3]),v2))
                den1 = den1 + np.multiply(v3[i3]**2,np.matmul(E[:,:,i3],v2*v2)).reshape(den1.shape[0],1)        
            v1 = V1[:,q].reshape(den1.shape[0],1)/den1
            v1 = v1/norm(v1)
            for i3 in range(n3):
                V2[:,q] = V2[:,q] + np.multiply(v3[i3],np.matmul((TE[:,:,i3]-A[:,:,i3]).T,v1)).flatten()
                den2 = den2 + np.multiply(v3[i3]**2,np.matmul(E[:,:,i3].T,np.multiply(v1,v1)))
            v2 = V2[:,q].reshape(den2.shape[0],1)/den2
            v2 = v2/norm(v2) 
            for i3 in range(n3):
                V3[i3,q] = (np.matmul(v1.T,np.matmul(TE[:,:,i3]-A[:,:,i3],v2))/np.matmul(np.matmul((v1*v1).T,(E[:,:,i3])),(v2*v2))).flatten()        
                # TODO: check for nans or zero estimates here 
            V1[:,q] = v1.flatten()
            V2[:,q] = v2.flatten()
            S[q] = norm(V3[:,q])
            V3[:,q] = V3[:,q]/norm(V3[:,q]) 
        ERR = TE - E*CPcomp(S,V1,V2,V3)
        normERR = 0  
        for i3 in range(n3):
            normERR = normERR + norm(ERR[:,:,i3])**2
        if np.sqrt(normERR/normTE) < tol: 
            break
    dist = np.sqrt(normERR/normTE)

    # reverse (in order of var and return)
    return V1[:,::-1], V2[:,::-1], V3[:,::-1], np.diag(S.flatten()), dist

def CPcomp(S,U1,U2,U3):
    ns, rs = S.shape
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    n3, r3 = U3.shape
    r = min([rs, r1, r2, r3])
    T = np.zeros((n1,n2,n3))
    for i in range(n3):
        t_i = np.diag(np.multiply(U3[i,:],S.T)[0])
        T[:,:,i] = np.matmul(np.matmul(U1,t_i),U2.T)
    return T

def TenProj(T, U1, U2, U3):
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    n3, r3 = U3.shape
    M = np.zeros((r1,r2,r3))
    for i in range(r3):
        A = np.zeros((n1,n2))
        for j in range(n3):
            A = A + T[:,:,j]*U3[j,i]
        M[:,:,i] = np.matmul(np.matmul(U1.T,A),U2)
    return M

def RTPM(T, max_iter):
    #RTPM
    n1, n2, n3 = T.shape
    u1 = randn(n1,1)/norm(randn(n1,1))
    u2 = randn(n2,1)/norm(randn(n2,1))
    u3 = randn(n3,1)/norm(randn(n3,1))
    #conv
    for itr in range(max_iter):
        v1 = np.zeros((n1,1))
        v2 = np.zeros((n2,1))
        v3 = np.zeros((n3,1))
        for i3 in range(n3):
            v3[i3] = np.matmul(np.matmul(u1.T,T[:,:,i3]),u2)
            v1 = v1 + np.matmul(u3[i3][0]*T[:,:,i3],u2)
            v2 = v2 + np.matmul(u3[i3][0]*T[:,:,i3].T,u1)
        u10 = u1
        u1 = v1/norm(v1)
        u20 = u2
        u2 = v2/norm(v2)
        u30 = u3
        u3 = v3/norm(v3)
        if(norm(u10-u1)+norm(u20-u2)+norm(u30-u3)<1e-7) :
            break
    return u1.flatten(),u2.flatten(),u3.flatten()
