"""qsvd-v4.py"""

# A = Original matrix
# B = Square root of Matrix A
# U, V, L = Matrices obtained after SVD
# Ut, Vt, Lt = Truncated U, V, L matrices
# Bnp = Matrix obtained from U, V, L with no phase added or truncation done
# Bp = Matrix obtained from U, V, L after truncation and phase added
# Bt = Matrix obtained from U, V, L after truncation, no phase added

""" Import Libraries """
import numpy as np
import scipy.optimize as optimize


""" Optimization Algorithm """
""" New Matrix """
def newMat_2(x, Ut,Lt,Vt):
    l0,l1 = Lt[0], Lt[1]
    U_new = np.zeros((Ut.shape), dtype=np.cfloat)
    for i in range(Ut.shape[0]):
      U_new[i][0] = np.cos(x[i]) / l0
      U_new[i][1] = (np.sin(x[i]) / l1) * np.exp(1j*x[i+Ut.shape[0]])
    return np.dot(np.dot(U_new,np.diag(Lt)), Vt)


def newMat_3(x, Ut,Lt,Vt):
    l0,l1,l2 = Lt[0], Lt[1], Lt[2]
    U_new = np.zeros((Ut.shape), dtype=np.cfloat)
    for i in range(Ut.shape[0]):
      U_new[i][0] = np.cos(x[i]) / l0
      U_new[i][1] = (np.sin(x[i]) / l1) * (np.cos(x[i+Ut.shape[0]])) * np.exp(1j*x[i+2*Ut.shape[0]])
      U_new[i][2] = (np.sin(x[i]) / l2) * (np.sin(x[i+Ut.shape[0]])) * np.exp(1j*x[i+3*Ut.shape[0]])
    return np.dot(np.dot(U_new,np.diag(Lt)), Vt)


def newMat_4(x, Ut,Lt,Vt):
    l0,l1,l2,l3 = Lt[0], Lt[1], Lt[2], Lt[3]
    U_new = np.zeros((Ut.shape), dtype=np.cfloat)
    for i in range(Ut.shape[0]):
      U_new[i][0] = (np.cos(x[i]) / l0) * (np.cos(x[i+Ut.shape[0]]))
      U_new[i][1] = (np.cos(x[i]) / l1) * (np.sin(x[i+Ut.shape[0]])) * np.exp(1j*x[i+3*Ut.shape[0]])
      U_new[i][2] = (np.sin(x[i]) / l2) * (np.cos(x[i+2*Ut.shape[0]])) * np.exp(1j*x[i+4*Ut.shape[0]])
      U_new[i][3] = (np.sin(x[i]) / l3) * (np.sin(x[i+2*Ut.shape[0]])) * np.exp(1j*x[i+5*Ut.shape[0]])
    return np.dot(np.dot(U_new,np.diag(Lt)), Vt)


""" Cost Function """
def costFn(x, Ut,Lt,Vt,B,k):
    if k==2:
      Bp = newMat_2(x, Ut, Lt, Vt)
    elif k==3:
      Bp = newMat_3(x, Ut, Lt, Vt)
    elif k==4:
      Bp = newMat_4(x, Ut, Lt, Vt)
    else:
      return 0
    loss = np.linalg.norm(B**2 - np.abs(Bp)**2)
    return (loss)


""" Calculate Results """
def calcResults(k, a=0, b=0, c=0, d=0):
  print("k = ", str(k))
  start1 = k+1 if a==0 else a
  end1 = k+9 if b==0 else b
  start2 = k+1 if c==0 else c
  end2 = k+9 if d==0 else d

  for m in range(start1, end1):
    for n in range(start2, end2):
      print("m = ",m,", n = ",n)
      filename= ('data/k{0}/{1}{2}.npy'.format(str(k),str(m),str(n)))
      res = np.zeros((100,2))
      for i in range(100):
          A = np.random.rand(m, n)
          for j in range(m): A[j] /= sum(A[j])
          B = np.sqrt(A)
          U, L, V = np.linalg.svd(B, full_matrices=False)
          initial_guess = np.ones((2*m*(k-1),), dtype=np.longdouble)
          Ut = U[:, :k]
          Vt = V[:k]
          Lt = L[:k]
          Bt = np.dot(np.dot(Ut,np.diag(Lt)), Vt)
          result = optimize.minimize(fun=costFn, x0=initial_guess, args=(Ut,Lt,Vt,B,k),
                                      tol=1e-7, method='Nelder-Mead', options={'maxiter':1e+10})
          res[i][0] = (np.linalg.norm(B**2 - Bt**2))
          res[i][1] = costFn(result.x,Ut,Lt,Vt,B,k)
          if(i%10==0):
            print(i, " ", end='')
      np.save(filename, res)
  return




""" Main Function """
def main():
  """ Save Results """ 
  """ k = 2 """ 
  #calcResults(2)
  """ k = 3 """ 
  calcResults(k=3, a=4, b=6, c=4, d=12)
  """ k = 4 """ 
  #calcResults(4)
  

if __name__=="__main__":
  main()