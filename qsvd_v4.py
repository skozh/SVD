# -*- coding: utf-8 -*-
"""qsvd-v4.py"""

# A = Original matrix
# B = Square root of Matrix A
# U, V, L = Matrices obtained after SVD
# Ut, Vt, Lt = Truncated U, V, L matrices
# Bnp = Matrix obtained from U, V, L with no phase added or truncation done
# Bp = Matrix obtained from U, V, L after truncation and phase added
# Bt = Matrix obtained from U, V, L after truncation, no phase added

## Import Libraries
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


"""Optimization Algorithm"""
"""New Matrix"""
def newMat(x, Ut,Lt,Vt):
    l0,l1 = Lt[0], Lt[1]
    U_new = np.zeros((Ut.shape), dtype=np.cfloat)
    for i in range(Ut.shape[0]):
      U_new[i][0] = np.cos(x[i]) / l0
      U_new[i][1] = (np.sin(x[i]) / l1) * np.exp(1j*x[i+Ut.shape[0]])
    return np.dot(np.dot(U_new,np.diag(Lt)), Vt)


"""Cost Function"""
def costFn(x, Ut,Lt,Vt,B):
    Bp = newMat(x, Ut,Lt,Vt)
    loss = np.linalg.norm(B**2 - np.abs(Bp)**2)
    return (loss)


""" Calculate Results"""
def calc(p, q):
  final_res = []
  for m in range(3, 11):
    for n in range(3, 11):
      
      print("m = ",m,", n = ",n)
      res = np.zeros((1000,2))
      for i in range(1000):
          A = np.random.rand(m, n)
          for j in range(m): A[j] /= sum(A[j])
          B = np.sqrt(A)
          U, L, V = np.linalg.svd(B, full_matrices=False)
          Ut = U[:,[0,1]]
          Vt = V[[0,1]]
          Lt = L[[0,1]]
          Bt = np.dot(np.dot(Ut,np.diag(Lt)), Vt)
          initial_guess = np.ones((m*2,), dtype=np.longdouble)
          result = optimize.minimize(fun=costFn, x0=initial_guess, args=(Ut,Lt,Vt,B),
                                      tol=1e-10, method='Nelder-Mead', options={'maxiter':1e+10})
          res[i][0] = (np.linalg.norm(B**2 - Bt**2))
          res[i][1] = costFn(result.x,Ut,Lt,Vt,B)
          if(i%100==0):
            print(i, " ", end='')

      # Save sample records
      if ((m==p) and (n==q)):
        df = pd.DataFrame(res, columns=['Initial', 'Final'])
        df['M'] = m
        df['N'] = n
        df['Diff'] = (df['Initial'] - df['Final'])
        df['RI'] = (df['Initial'] - df['Final'])*100/df['Initial']
        df = df[['M', 'N', 'Initial', 'Final', 'Diff', 'RI']]
        df.to_csv('sample_out.csv', sep=',', index=False)

      mean_initial, mean_final = res.mean(axis=0)
      ri_mean= ((mean_initial - mean_final)*100/mean_initial)
      ri_std = np.mean(np.std(res[:,0]-res[:,1]))
      data = {"m": m, "n": n, 
              "mean_initial_dist": mean_initial, "mean_final_dist": mean_final,
              "mean_RI": ri_mean, "std_RI": ri_std}
      final_res.append(data)
      print("\n")
  final_arr = pd.DataFrame.from_dict(final_res)
  return final_arr


""" Main Function """
def main():

  # Save Results
  #final_arr = calc(7, 5)
  #final_arr.to_pickle("./final_arr.pkl")

  # Load Results
  final_arr = pd.read_pickle("./final_arr.pkl")

  print("Results")
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
  plt.savefig('plot.png')
  plt.show()


if __name__=="__main__":
  main()