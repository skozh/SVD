﻿// See https://aka.ms/new-console-template for more information
using System.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using mathnet = MathNet.Numerics.LinearAlgebra;

class qsvd
{
    static Matrix<Complex> newMatrix(mathnet.Vector<double> x, Matrix<double> Ut, 
                                        Matrix<double> Vt, Matrix<double> Wt, int k){
        double l0, l1, l2, l3;
        Complex im = new Complex(0,1);
        Matrix<Complex> U_new = Matrix<Complex>.Build.Dense(Ut.RowCount, Ut.ColumnCount);
        switch(k){
            case 2:
                l0=Wt[0,0];
                l1=Wt[1,1];
                for (int i=0; i<Ut.RowCount; i++){
                    U_new[i,0] = Math.Cos(x[i])/l0;
                    U_new[i,1] = Complex.Multiply((Math.Sin(x[i])/l1), Complex.Exp(Complex.Multiply(im,x[i+Ut.RowCount])));
                }
                break;
            case 3:
                l0=Wt[0,0];
                l1=Wt[1,1];
                l2=Wt[2,2];
                for (int i=0; i<Ut.RowCount; i++){
                    U_new[i,0] = Math.Cos(x[i])/l0;
                    U_new[i,1] = Complex.Multiply((Math.Sin(x[i])/l1) * Math.Cos(x[i+Ut.RowCount]), Complex.Exp(Complex.Multiply(im,x[i+2*Ut.RowCount])));
                    U_new[i,2] = Complex.Multiply((Math.Sin(x[i])/l2) * Math.Sin(x[i+Ut.RowCount]), Complex.Exp(Complex.Multiply(im, x[i+3*Ut.RowCount])));
                }
                break;
            case 4:
                l0=Wt[0,0];
                l1=Wt[1,1];
                l2=Wt[2,2];
                l3=Wt[3,3];
                for (int i=0; i<Ut.RowCount; i++){
                    U_new[i,0] = (Math.Cos(x[i])/l0) * (Math.Cos(x[i+Ut.RowCount]));
                    U_new[i,1] = Complex.Multiply((Math.Cos(x[i])/l1) * (Math.Sin(x[i+Ut.RowCount])), Complex.Exp(Complex.Multiply(im,x[i+3*Ut.RowCount])));
                    U_new[i,2] = Complex.Multiply((Math.Sin(x[i])/l2) * (Math.Cos(x[i+2*Ut.RowCount])), Complex.Exp(Complex.Multiply(im,x[i+4*Ut.RowCount])));
                    U_new[i,3] = Complex.Multiply((Math.Sin(x[i])/l3) * (Math.Sin(x[i+2*Ut.RowCount])), Complex.Exp(Complex.Multiply(im,x[i+5*Ut.RowCount])));
                }
                break;
        }
        Matrix<Complex> Bp = U_new.Multiply(Wt.ToComplex()).Multiply(Vt.ToComplex());
        return Bp;
    }


    static double costFunction(mathnet.Vector<double> x, Matrix<double> Ut, 
                                Matrix<double> Vt, Matrix<double> Wt, Matrix<double> B, int k){
        double loss = 0;
        Matrix<Complex> Bp = newMatrix(x, Ut, Vt, Wt, k);
        loss = ((B.ToComplex().Map(c => Complex.Multiply(Complex.Conjugate(c),c).Real)) 
                    - (Bp.Map(c => Complex.Multiply(Complex.Conjugate(c),c).Real))).FrobeniusNorm(); 
        return loss;
    }


    static void calcResults(int k){
        int start = k+1;
        int end = k+9;
        Parallel.For(start, end, m => {
            Parallel.For(start, end, n => {
                double[,] res= new double[10,2];
                for (int i=0; i<10; i++){

                    //Calculate SVD
                    Matrix<double> A = Matrix<double>.Build.Random(m, n, new ContinuousUniform(0,1));
                    Matrix<double> B = A.Map(c => Math.Sqrt(c));
                    var svd = B.Svd(true);
                    Matrix<double> B_org = svd.U * svd.W * svd.VT;
                    
                    //Truncated SVD
                    Matrix<double> Ut = svd.U.SubMatrix(0,m,0,k);
                    Matrix<double> Vt = svd.VT.SubMatrix(0,k,0,n);
                    Matrix<double> Wt = svd.W.SubMatrix(0,k,0,k);
                    Matrix<double> Bt = Ut * Wt * Vt;

                    //Loss 1
                    res[i,0] = (B.ToComplex().Map(c => Complex.Multiply(Complex.Conjugate(c),c).Real)
                                -Bt.ToComplex().Map(c => Complex.Multiply(Complex.Conjugate(c),c).Real)).FrobeniusNorm();

                    //Optimization
                    int size = 2*m*(k-1);
                    mathnet.Vector<double> initial_guess = 0.9*mathnet.Vector<double>.Build.Dense(size);
                    var f1 = new Func<mathnet.Vector<double>, double>(x => costFunction(x, Ut, Vt, Wt, B, k));
                    var obj = ObjectiveFunction.Value(f1);
                    var solver = new NelderMeadSimplex(1e-20, 100000);

                    //Loss 2                    
                    try{
                        var result= solver.FindMinimum(obj, initial_guess);
                        res[i,1] = costFunction(result.MinimizingPoint, Ut, Vt, Wt, B, k);
                    }
                    catch{
                        res[i,1] = costFunction(initial_guess, Ut, Vt, Wt, B, k);
                    }
                    Console.WriteLine("M={0}, N={1}, Thread={2}", m, n, Thread.CurrentThread.ManagedThreadId);
                    Console.WriteLine(res[i,0].ToString() +" --> "+ res[i,1].ToString());
                }
            });
        });   
    }


    static void Main(string[] args){
        calcResults(2);  
    }
}