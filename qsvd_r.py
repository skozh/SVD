# %%
import numpy as np
import scipy.optimize as optimize
from concurrent.futures import ProcessPoolExecutor


""" Optimization Algorithm """
""" New Matrix """
def newMat(x, Vt, k):
  V_new = np.zeros((Vt.shape), dtype=np.cfloat)
  if k==2:
    V_new[0] = np.cos(x[0])
    V_new[1] = (np.sin(x[0])) 
  elif k==3:
    V_new[0] = np.cos(x[0])
    V_new[1] = (np.sin(x[0])) * (np.cos(x[1])) 
    V_new[2] = (np.sin(x[0])) * (np.sin(x[1])) 
  elif k==4:
    V_new[0] = (np.cos(x[0])) * (np.cos(x[1]))
    V_new[1] = (np.cos(x[0])) * (np.sin(x[1])) 
    V_new[2] = (np.sin(x[0])) * (np.cos(x[2])) 
    V_new[3] = (np.sin(x[0])) * (np.sin(x[2])) 
  elif k==5:
    V_new[0] = (np.cos(x[0])) * (np.cos(x[1]))
    V_new[1] = (np.cos(x[0])) * (np.sin(x[1])) 
    V_new[2] = (np.sin(x[0])) * (np.cos(x[2])) 
    V_new[3] = (np.sin(x[0])) * (np.sin(x[2])) * (np.sin(x[3])) 
    V_new[4] = (np.sin(x[0])) * (np.sin(x[2])) * (np.cos(x[3])) 
  else:
    V_new[0] = (np.cos(x[0])) * (np.cos(x[1]))
    V_new[1] = (np.cos(x[0])) * (np.sin(x[1])) 
    V_new[2] = (np.sin(x[0])) * (np.cos(x[2])) 
    V_new[3] = (np.sin(x[0])) * (np.sin(x[2])) * (np.sin(x[3])) 
    V_new[4] = (np.sin(x[0])) * (np.sin(x[2])) * (np.cos(x[3])) 
    V_new[5] = (np.sin(x[0])) * (np.sin(x[2])) * (np.cos(x[3])) * (np.sin(x[4]))
  return V_new


""" Cost Function """
def costFn(x, Ut, Vt, A, k):
    V_new = newMat(x, Vt, k)
    Bp = np.dot(Ut, V_new) 
    loss = np.linalg.norm(A - Bp*np.conjugate(Bp))
    return (loss)


def SVD(mn_comb):
    m, n, k = mn_comb;
    print ("k = ",k,"m = ",m,", n = ",n)
    filename= ('data/k{0}_r/m{1}_n{2}.npy'.format(str(k),str(m),str(n)))
    res = np.zeros((100,2))
    for i in range(100):
        A = np.random.rand(m, n)
        A = A/A.sum(axis=0)         # Optimize column-wise

        # Classic Truncated SVD
        U, L, V = np.linalg.svd(A, full_matrices=False)
        Ut = U[:, :k]
        Vt = V[:k]
        Lt = L[:k]
        At = np.dot(np.dot(Ut,np.diag(Lt)), Vt)
        res[i][0] = (np.linalg.norm(A - At))

        # New SVD with Real V
        B = np.sqrt(A)
        U, L, V = np.linalg.svd(B, full_matrices=False)
        
        Ut = U[:, :k]
        Vt = V[:k]
        Lt = L[:k]
        initial_guess = np.ones(((k-1),), dtype=np.longdouble)
        V_new = np.zeros(Vt.shape, dtype=np.cfloat)
        for col in range(Vt.shape[1]):
            result = optimize.minimize(fun=costFn, x0=initial_guess, args=(Ut,Vt[:, col],A[:,col],k),
                                    tol=1e-7, method='Nelder-Mead', options={'maxiter':1e+10})
            V_new[:,col] = newMat(result.x, Vt[:, col], k)
        Bp = np.dot(Ut, V_new)  
        res[i][1] = (np.linalg.norm(A - np.conjugate(Bp)*Bp))

        if i%10==0: print(i, end=' ')
    np.save(filename, res)
    return filename


def calcResults(k):
    m = np.arange(k+1, k+9)
    n = np.arange(k+1, k+9)
    mn_comb = [(x, y, k) for x in m for y in n]
    with ProcessPoolExecutor(max_workers=2) as executor:
        for r in executor.map(SVD, mn_comb):
            print("File {} saved!".format(r))
    return None


def main():
  calcResults(k=4)


if __name__ == "__main__":
  main()