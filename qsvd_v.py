# %%
import numpy as np
import scipy.optimize as optimize
from concurrent.futures import ProcessPoolExecutor


def findCombo():
  combo = []
  for k1 in range(2, 12):
      for m in range(k1+1, k1+9):
          for n in range(k1+1, k1+9):
              k2 = (k1*(m+n-1)+2*n)/(2*n+m-1)
              if (k2%1==0):
                  if [k1, int(k2), m, n] not in combo: 
                      combo.append([k1, int(k2), m, n])
  return np.array(combo, dtype=object)


""" Optimization Algorithm """
""" New Matrix """
def newMat(x, Vt, k):
  V_new = np.zeros((Vt.shape), dtype=np.cfloat)
  if k==2:
    V_new[0] = np.cos(x[0])
    V_new[1] = (np.sin(x[0])) * np.exp(1j*x[1])
  elif k==3:
    V_new[0] = np.cos(x[0])
    V_new[1] = (np.sin(x[0])) * (np.cos(x[1])) * np.exp(1j*x[2])
    V_new[2] = (np.sin(x[0])) * (np.sin(x[1])) * np.exp(1j*x[3])
  elif k==4:
    V_new[0] = (np.cos(x[0])) * (np.cos(x[1]))
    V_new[1] = (np.cos(x[0])) * (np.sin(x[1])) * np.exp(1j*x[3])
    V_new[2] = (np.sin(x[0])) * (np.cos(x[2])) * np.exp(1j*x[4])
    V_new[3] = (np.sin(x[0])) * (np.sin(x[2])) * np.exp(1j*x[5])
  elif k==5:
    V_new[0] = (np.cos(x[0])) * (np.cos(x[1]))
    V_new[1] = (np.cos(x[0])) * (np.sin(x[1])) * np.exp(1j*x[3])
    V_new[2] = (np.sin(x[0])) * (np.cos(x[2])) * np.exp(1j*x[4])
    V_new[3] = (np.sin(x[0])) * (np.sin(x[2])) * (np.sin(x[6])) * np.exp(1j*x[5])
    V_new[4] = (np.sin(x[0])) * (np.sin(x[2])) * (np.cos(x[6])) * np.exp(1j*x[7])
  else:
    V_new[0] = (np.cos(x[0])) * (np.cos(x[1]))
    V_new[1] = (np.cos(x[0])) * (np.sin(x[1])) * np.exp(1j*x[3])
    V_new[2] = (np.sin(x[0])) * (np.cos(x[2])) * np.exp(1j*x[4])
    V_new[3] = (np.sin(x[0])) * (np.sin(x[2])) * (np.sin(x[6])) * np.exp(1j*x[5])
    V_new[4] = (np.sin(x[0])) * (np.sin(x[2])) * (np.cos(x[6])) * np.exp(1j*x[7])
    V_new[5] = (np.sin(x[0])) * (np.sin(x[2])) * (np.cos(x[6])) * (np.sin(x[8])) * np.exp(1j*x[9])
  return V_new


""" Cost Function """
def costFn(x, Ut, Vt, A, k):
    V_new = newMat(x, Vt, k)
    Bp = np.dot(Ut, V_new) 
    loss = np.linalg.norm(A - Bp*np.conjugate(Bp))
    return (loss)


def SVD(p):
    [k, k_new, m, n] = p
    print ("m = ",m,", n = ",n)
    res = np.zeros((100,3))
    for i in range(100):
        A = np.random.rand(m, n)
        A = A/A.sum(axis=0)         # Optimize column-wise

        #Classic Truncated SVD
        U, L, V = np.linalg.svd(A, full_matrices=False)
        Ut = U[:, :k]
        Vt = V[:k]
        Lt = L[:k]
        At = np.dot(np.dot(Ut,np.diag(Lt)), Vt)
        res[i][0] = (np.linalg.norm(A - At))

        # Complex SVD
        B = np.sqrt(A)
        U, L, V = np.linalg.svd(B, full_matrices=False)
        # Complex SVD with k
        if (k<=6):                                      # Skip when k>6
            Ut = U[:, :k]
            Vt = V[:k]
            Lt = L[:k]
            initial_guess = np.ones((2*(k-1),), dtype=np.longdouble)
            V_new = np.zeros(Vt.shape, dtype=np.cfloat)
            for col in range(Vt.shape[1]):
                result = optimize.minimize(fun=costFn, x0=initial_guess, args=(Ut,Vt[:, col],A[:,col],k),
                                        tol=1e-7, method='Nelder-Mead', options={'maxiter':1e+10})
                V_new[:,col] = newMat(result.x, Vt[:, col], k)
            Bp = np.dot(Ut, V_new)  
            res[i][1] = (np.linalg.norm(A - np.conjugate(Bp)*Bp))
        else:
            res[i][1] = 0

        # Complex SVD with k_new
        Ut = U[:, :k_new]
        Vt = V[:k_new]
        Lt = L[:k_new]
        initial_guess = np.ones((2*n*(k_new-1),), dtype=np.longdouble)
        V_new = np.zeros(Vt.shape, dtype=np.cfloat)
        for col in range(Vt.shape[1]):
            result = optimize.minimize(fun=costFn, x0=initial_guess, args=(Ut,Vt[:, col],A[:,col],k_new),
                                    tol=1e-7, method='Nelder-Mead', options={'maxiter':1e+10})
            V_new[:,col] = newMat(result.x, Vt[:, col], k_new)
        Bp = np.dot(Ut, V_new)  
        res[i][2] = (np.linalg.norm(A - np.conjugate(Bp)*Bp))

        if i%10==0: print(i, end=' ')
    print([m, n, *res.mean(axis=0)])
    print('\n')
    return ([m, n, *res.mean(axis=0)])


def calcResults(mn_comb):
    final_res = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for r in executor.map(SVD, mn_comb):
            final_res.append(r)
    return final_res


def main():
  npc = findCombo()
  #print(npc)

  #mn_comb = npc[npc[:,0]==5] # k = 5
  #mn_comb = npc[npc[:,0]==7] # k = 7
  mn_comb = npc[npc[:,0]==8] # k = 8
  res = calcResults(mn_comb)
  print(res)


if __name__ == "__main__":
  main()