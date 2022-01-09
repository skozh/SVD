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
import pandas as pd
import scipy.optimize as optimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
      U_new[i][1] = (np.sin(x[i]) / l1) * (np.cos(x[i+Ut.shape[0]]) / l1) * np.exp(1j*x[i+2*Ut.shape[0]])
      U_new[i][2] = (np.sin(x[i]) / l2) * (np.sin(x[i+Ut.shape[0]]) / l2) * np.exp(1j*x[i+2*Ut.shape[0]])
    return np.dot(np.dot(U_new,np.diag(Lt)), Vt)


def newMat_4(x, Ut,Lt,Vt):
    l0,l1,l2,l3 = Lt[0], Lt[1], Lt[2], Lt[3]
    U_new = np.zeros((Ut.shape), dtype=np.cfloat)
    for i in range(Ut.shape[0]):
      U_new[i][0] = (np.cos(x[i]) / l0) * (np.cos(x[i+Ut.shape[0]]) / l0)
      U_new[i][1] = (np.cos(x[i]) / l1) * (np.sin(x[i+Ut.shape[0]]) / l1) * np.exp(1j*x[i+2*Ut.shape[0]])
      U_new[i][2] = (np.sin(x[i]) / l2) * (np.cos(x[i+Ut.shape[0]]) / l2) * np.exp(1j*x[i+3*Ut.shape[0]])
      U_new[i][3] = (np.sin(x[i]) / l3) * (np.sin(x[i+Ut.shape[0]]) / l3) * np.exp(1j*x[i+3*Ut.shape[0]])
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
def calcResults(p, q, k):
  final_res = []
  filename = "sample_out_" +str(k)+".csv"
  
  print("k = ", str(k))
  for m in range(k+1, 11):
    for n in range(k+1, 11):
      print("m = ",m,", n = ",n)
      res = np.zeros((10,2))
      for i in range(10):
          A = np.random.rand(m, n)
          for j in range(m): A[j] /= sum(A[j])
          B = np.sqrt(A)
          U, L, V = np.linalg.svd(B, full_matrices=False)
          initial_guess = np.ones((m*k,), dtype=np.longdouble)
          Ut = U[:, :k]
          Vt = V[:k]
          Lt = L[:k]
          Bt = np.dot(np.dot(Ut,np.diag(Lt)), Vt)
          result = optimize.minimize(fun=costFn, x0=initial_guess, args=(Ut,Lt,Vt,B,k),
                                      tol=1e-10, method='Nelder-Mead', options={'maxiter':1e+10})
          res[i][0] = (np.linalg.norm(B**2 - Bt**2))
          res[i][1] = costFn(result.x,Ut,Lt,Vt,B,k)
          #if(i%10==0):
          print(i, " ", end='')

      """ Save sample records """
      if ((m==p) and (n==q)):
        df = pd.DataFrame(res, columns=['Initial', 'Final'])
        df['M'] = m
        df['N'] = n
        df['Diff'] = (df['Initial'] - df['Final'])
        df['RI'] = (df['Initial'] - df['Final'])*100/df['Initial']
        df = df[['M', 'N', 'Initial', 'Final', 'Diff', 'RI']]
        df.to_csv(filename, sep=',', index=False)

      mean_initial, mean_final = res.mean(axis=0)
      ri_mean= ((mean_initial - mean_final)*100/mean_initial)
      ri_std = np.mean(np.std(res[:,0]-res[:,1]))
      data = {"M": m, "N": n, 
              "mean_initial_dist": mean_initial, "mean_final_dist": mean_final,
              "mean_RI": ri_mean, "std_RI": ri_std}
      final_res.append(data)
      print("\n")
  final_arr = pd.DataFrame.from_dict(final_res)
  return final_arr


""" Show Results """
def printResults(final_arr, k):

  print("Results for k="+str(k))
  print("\nMean Values: \n", final_arr[['mean_RI', 'std_RI']].mean())
  print("\nMin Values: \n", final_arr[['mean_RI', 'std_RI']].min())
  print("\nMax Values: \n", final_arr[['mean_RI', 'std_RI']].max())
  print("\nMax Entry: \n", final_arr.iloc[final_arr['mean_RI'].idxmax()])
  print("\nMin Entry: \n", final_arr.iloc[final_arr['mean_RI'].idxmin()])

  """ Plot """
  fig = plt.figure(figsize=(15,10))
  ax = plt.axes(projection='3d')
  ax.plot_trisurf(final_arr['M'], final_arr['N'], final_arr['mean_RI'],cmap=plt.cm.inferno)
  ax.set_xlabel('M')
  ax.set_ylabel('N')
  ax.set_zlabel('RI%')
  filename = "plot_"+str(k)+".png"
  plt.savefig(filename)
  plt.show()


""" Main Function """
def main():
  """ Save Results """ 
  """ k = 2 """ 
  final_arr_2 = calcResults(7, 5, 2)
  final_arr_2.to_pickle("./final_arr_2.pkl")
  """ k = 3 """ 
  final_arr_3 = calcResults(7, 5, 3)
  final_arr_3.to_pickle("./final_arr_3.pkl")
  """ k = 4 """ 
  final_arr_4 = calcResults(7, 5, 4)
  final_arr_4.to_pickle("./final_arr_4.pkl")

  """ Load Results """
  final_arr_2 = pd.read_pickle("./final_arr_2.pkl")
  final_arr_3 = pd.read_pickle("./final_arr_3.pkl")
  final_arr_4 = pd.read_pickle("./final_arr_4.pkl")

  """ Print Results """ 
  printResults(final_arr_2, k=2)
  printResults(final_arr_3, k=3)
  printResults(final_arr_4, k=4)
  

if __name__=="__main__":
  main()