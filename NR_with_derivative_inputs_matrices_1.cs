using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Xml.Linq;
using System.Globalization;

namespace NR_with_derivative_inputs_matrices
{
    class Program
    {
        // Taking the inputs from linear and quadratic matrices
        public static double[][] readEquationCoefficient(string filepath)
        {
            Console.WriteLine();
            string[] lines = File.ReadAllLines(filepath);
            // No. of equations 
            int n = Convert.ToInt32(lines[0]);
            Console.WriteLine("Number of " + filepath + " - " + n + "\n");
            // No. of variables
            int n_var = Convert.ToInt32(lines[1]);
            Console.WriteLine("Number of variable in each equation - " + n_var + "\n");

            // Taking coefficient of linear equations from the user
            double[][] coefficient = new double[n][];
            Console.WriteLine("The constant and coefficient of " + filepath + " - ");
            for (int i = 0; i < n; i++)
            {
                string line = lines[i + 2];
                string[] values = line.Trim().Split(' ');
                coefficient[i] = new double[values.Length];
                for (int j = 0; j < values.Length; j++)
                {
                  coefficient[i][j] = Convert.ToDouble(values[j]);   
                    Console.Write(coefficient[i][j] + " ");
                }
                Console.WriteLine();
            }
            return coefficient;
        }
        // Function for skipping the 3rd row from the input quadratic matrix
        public static double[][] quadraticCoefficient(double[][] quadratic_eq)
        {
            double[][] quadratic_coefficient = new double[quadratic_eq.Length - 1][];  // takes only 7 values in column
            for (int i = 0; i < quadratic_eq.Length; i++)
            {
                if (i < 2)
                {
                    quadratic_coefficient[i] = new double[16];
                    for (int j = 11; j < quadratic_eq[0].Length; j++)
                    {
                        quadratic_coefficient[i][j - 11] = quadratic_eq[i][j];
                    }
                }
                else if (i > 2)
                {
                    quadratic_coefficient[i - 1] = new double[16];
                    for (int j = 11; j < quadratic_eq[0].Length; j++)
                    {
                        quadratic_coefficient[i - 1][j - 11] = quadratic_eq[i][j];
                    }
                }
            }
            return quadratic_coefficient;
        }
        public static void printingArray1D(double[] inputs, string text)
        {
            Console.WriteLine();
            Console.WriteLine(text + "are : - ");
            for (int i = 0; i < inputs.Length; i++)
            {
                Console.Write(inputs[i] + " ");
            }
            Console.WriteLine();
        }
        public static void printingArray2D(double[][] input, string text)
        {
            Console.WriteLine();
            Console.WriteLine(text + " Are : - ");
            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < input[0].Length; j++)
                {
                    Console.Write(input[i][j] + " ");
                }
                Console.WriteLine();
            }
        }
        public static double[] calculateFValues(double[] x_initial, double[] e, double[] d, int n)
        {
            double nu = 0.00001307;
            double pi = Math.PI;

            // Formula Used:
            // f[i] = -2.0 log(e[i]/3.7d[i] - 5.02*pi*d[i]*nu/4x[i] log(e[i]/3.7d[i] - 5.02*pi*d[i]*nu/4x[i] log(e[i]/3.7d[i] + 13*pi*d[i]*nu/4x[i])))]^(-2)

            double[] f = new double[n];
            for (int i = 0; i < n; i++)
            {
                double c1 = e[i] / (3.7 * d[i]);
                double c2 = (pi * d[i] * nu) / (4 * x_initial[i]);
                f[i] = -2 * (Math.Log10(c1 - 5.02 * c2 * Math.Log10(c1 - 5.02 * c2 * Math.Log(c1 + 13 * c2))));
                f[i] = Math.Pow(f[i], -2);
            }
            return f;
        }
        public static double[] calculateKValues(double[] f, double[] l, double[] rb, double[] kb45, double[] kb90,
            double[] n1, double[] n2, double[] d, int n)
        {
            double[] k = new double[n];
            for (int i = 0; i < n; i++)
            {
                double pi = Math.PI;
                double c1 = 0.08271;
                double c2 = Math.Pow(d[i], 4);
                double c3 = f[i] * (pi * (rb[i] / 2)) / d[i];

                double eq = 0;
                if (i > 0)
                {
                    double num = Math.Pow(d[i], 2);
                    double den = Math.Pow(d[i - 1], 2);
                    double division = num / den;
                    eq = Math.Pow((division - 1), 2);
                }

                k[i] = (c1 / c2) * (f[i] * (l[i] / d[i]) + (c3 + kb90[i]) * n1[i] + (c3 + kb45[i]) * n2[i] + eq);
            }
            return k;
        }
        // In the Main program:
        // 1) Taking the inputs from linear and quadratic matrices
        // 2) Making the derivative matrices for linear and quadratic inputs which is used in NR method
        static void Main(string[] args)
        {
            // Getting Linear Equation Coefficient
             double[][] linear_eq = readEquationCoefficient(@"linearInput.txt");
            /* =============================================*/
            // 1) Making the derivative for linear equations
            /* ============================================ */
            // Skipping the first column from represents the constant value
            double[][] linear_eq_skip = new double[linear_eq.Length][];
                 for(int i = 0; i < linear_eq.Length; i++) 
                   {
                     linear_eq_skip[i] = new double[linear_eq[i].Length - 1];
                     for(int j = 0; j < linear_eq[i].Length -1; j++) 
                        {
                           linear_eq_skip[i][j] = linear_eq[i][j+1];
                        }
                   }
            double[][] linear_derivative = new double[7][];
            for(int i = 0; i < linear_eq_skip.Length; i++)
            {
                linear_derivative[i] = new double[linear_eq_skip[i].Length + 7];
                for (int j = 0; j < 14; j++)
                {
                    if (j > 7)
                        linear_derivative[i][j] = 0;
                    else if (j < 7)
                        linear_derivative[i][j] = linear_eq_skip[i][j];
                }
            }
           // printingArray2D(linear_derivative, "linear Eqn derivative Coefficient");
            /* =============================================*/
            // 2) Making the derivative for quadratic equations
            /* ============================================ */
            double[][] full_quadratic_eq = readEquationCoefficient(@"quadraticInputheadloss.txt");
            printingArray2D(full_quadratic_eq, "readed coeff's of full Quadratic Matrix");
            // Getting Quadratic Equations Coefficient
             double[][] reduced_quadratic_eq = new double[7][];
            reduced_quadratic_eq = quadraticCoefficient(full_quadratic_eq);
            printingArray2D(reduced_quadratic_eq, "Reduced Quadratic Eqn Coefficient");
            // Skipping the first column from represents the constant value
            double[][] quadratic_eq_skip = new double[linear_eq.Length][];
            for (int i = 0; i < linear_eq.Length; i++)
            {
                quadratic_eq_skip[i] = new double[15]; // it defines the number of elements required in each row
                for (int j = 0; j < 15; j++)
                {
                    quadratic_eq_skip[i][j] = reduced_quadratic_eq[i][j + 1];
                }
            }
            //Console.WriteLine("Length of reduced quadratic equation  " + reduced_quadratic_eq[0].Length);
            printingArray2D(quadratic_eq_skip, "1st Skipped Quadratic Eqn Coefficient");
            // Skipping the 8th colummn
            double[][] quadratic_eq_skip_2 = new double[linear_eq.Length][];
            for (int i = 0; i < linear_eq.Length; i++)
            {
                quadratic_eq_skip_2[i] = new double[14]; // it defines the number of elements required in each row
                for (int j = 0; j < reduced_quadratic_eq[0].Length -1; j++) // elements in each row of reduced quadratic is 16
                {
                    if (j < 7)
                        quadratic_eq_skip_2[i][j] = quadratic_eq_skip[i][j];
                    else if (j > 7)
                        quadratic_eq_skip_2[i][j-1] = quadratic_eq_skip[i][j];
                }
            }
            double[][] quadratic_derivative = new double[linear_eq.Length][];
            quadratic_derivative = quadratic_eq_skip_2;
            printingArray2D(quadratic_derivative, "derivative Quadratic Eqn Coefficient");
            // ============== Computing f values =================
            double[] x_initial_1 = new double[full_quadratic_eq.Length];
            double[] x_initial_2 = new double[full_quadratic_eq.Length];
            double[] e = new double[full_quadratic_eq.Length];
            double[] d = new double[full_quadratic_eq.Length];
            double[] l = new double[full_quadratic_eq.Length];
            double[] rb = new double[full_quadratic_eq.Length];
            double[] kb45 = new double[full_quadratic_eq.Length];
            double[] kb90 = new double[full_quadratic_eq.Length];
            double[] n1 = new double[full_quadratic_eq.Length];
            double[] n2 = new double[full_quadratic_eq.Length];
            for(int i = 0; i < full_quadratic_eq.Length; i++)
            {
                l[i] = full_quadratic_eq[i][0];
                d[i] = full_quadratic_eq[i][1];
                e[i] = full_quadratic_eq[i][2];
                n1[i] = full_quadratic_eq[i][3];
                n2[i] = full_quadratic_eq[i][4];
                kb90[i] = full_quadratic_eq[i][5];
                kb45[i] = full_quadratic_eq[i][6];
                rb[i] = full_quadratic_eq[i][7];
                x_initial_1[i] = full_quadratic_eq[i][8];
                x_initial_2[i] = full_quadratic_eq[i][10];
            }
             double[] f = calculateFValues(x_initial_1, e, d, linear_eq.Length);
             printingArray1D(f, "F Values");
            // Computing k values
             double[] k = calculateKValues(f, l, rb, kb45, kb90, n1, n2, d, linear_eq.Length);
            double[] k_new = new double[6]; 
            for(int i = 0; i < linear_eq.Length; i++)
            {
                if (i < 2)
                    k_new[i] = k[i];
                else if (i > 2)
                    k_new[i - 1] = k[i];
            }
            printingArray1D(k, "Old K Values");
            printingArray1D(k_new, "New K Values");
            // Multiplication of quadratic derivative coefficients with the value of K needs to be done 
            for (int i = 0; i< quadratic_derivative.Length-1; i++)
            {
                for (int j = 0; j < quadratic_derivative[0].Length; j++)
                {
                    if (j > 6)
                        quadratic_derivative[i][j] = quadratic_derivative[i][j] * k_new[i];
                     else
                        quadratic_derivative[i][j] = quadratic_derivative[i][j];
                }
            }
            printingArray2D(quadratic_derivative, "derivative Quadratic Eqn Coefficient");       
            // initial = initial values for linear equations (x1 to x7) and for quadratic equations (x8 to x14)  
            // need to be taken from input linear matrix
            double[] initial_1 = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }; 
            // Need to be taken from input quadratic matrix
            double[] initial_2 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }; 
            // need to be taken from input linear and quadratic matrix
            double[] vector = { -0.1, 0.0, 0.1, -0.1, 0.0, 5.0, 1.5, 4.5, -2.0, 5.5, -4.0, -2.0, 0.0, 0.0};
            //   string filepath = @"C:\Users\sja\Documents\Projects_Csharp\NR_with_derivative_inputs_matrices\NR_with_derivative_inputs_matrices\bin\Debug\output.txt";
            /*   if (File.Exists(filepath))
               {
                   File.Delete(filepath);
               }

               NewtonRaphson(linear_derivative, quadratic_derivative, vector, 10, initial_1, initial_2);*/
            Console.ReadLine();
        }

       /* static void NewtonRaphson(double[][] linear_derivative, double[][] quadratic_derivative, double[] vector, int iterations, double[] initial_1, double[] initial_2)
        {
            // Base Method
            if (iterations == 0)
            {
                return;
            }
            string filepath = @"C:\Users\sja\Documents\Projects_Csharp\NR_with_derivative_inputs_matrices\NR_with_derivative_inputs_matrices\bin\Debug\output.txt";
            double[][] der_quadratic_eq = new double[quadratic_derivative.Length][];
            // Multiplying the input quadratic eqn with initial values
            // Need to define two different array of initial arrays, because we need initial1 and initial 2 in quadratic equations
            // 1) One for linear equations => initial 1
            // 2) Other for quadratic equations => initial 2

            for(int i = 0; i < quadratic_derivative.Length; i++ )
            {
                der_quadratic_eq[i] = new double[quadratic_derivative[i].Length];
                for(int j = 0; j < quadratic_derivative.Length; j++)
                {
                    if (j >= 7)
                            der_quadratic_eq[i][j] = 2* quadratic_derivative[i][j] * initial_1[j];  
                    else if( j < 7)
                        der_quadratic_eq[i][j] = quadratic_derivative[i][j] * initial_2[j];
                }
            }

            // Step 8: Merging the linear equation eq and quadratic eq together 






            // Step 9: Taking the inverse of merged matrix using MatrixInverse function





            // Step 10: Developing vector function to be multiplied with inverse matrix
                  // - for linear equations



                  // - for quadratic equations



            // Step 11: Multiplying inverse matrix with the vector





            // Step 12: Doing substraction of initial values with above multiplication result to have the value for the next iteration





            // Step 13: Writing of result in output file




        }*/
        //=====================================================================================
        static double[] MatrixVectorProduct(double[][] matrix, double[] vector)
        {
            // result of multiplying an n x m matrix by a m x 1 column vector (yielding an n x 1 column vector)
            int mRows = matrix.Length; int mCols = matrix[0].Length;
            int vRows = vector.Length;
            if (mCols != vRows)
                throw new Exception("Non-conformable matrix and vector in MatrixVectorProduct");
            double[] result = new double[mRows]; // an n x m matrix times a m x 1 column vector is a n x 1 column vector
            for (int i = 0; i < mRows; ++i)
                for (int j = 0; j < mCols; ++j)
                    result[i] += matrix[i][j] * vector[j];
            return result;
        }
        //======================================================================================
        // Creating the matrix 
        static double[][] MatrixCreate(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
        //=======================================================================================
        static double[][] MatrixIdentity(int n)
        {
            // return an n x n Identity matrix
            double[][] result = MatrixCreate(n, n);
            for (int i = 0; i < n; ++i)
                result[i][i] = 1.0;

            return result;
        }
        //========================================================================================
        static string MatrixAsString(double[][] matrix)
        {
            string s = "";
            for (int i = 0; i < matrix.Length; ++i)
            {
                for (int j = 0; j < matrix[i].Length; ++j)
                    s += matrix[i][j].ToString("F3").PadLeft(8) + " ";
                s += Environment.NewLine;
            }
            return s;
        }
        //==========================================================================================
        static bool MatrixAreEqual(double[][] matrixA, double[][] matrixB, double epsilon)
        {
            // true if all values in matrixA == corresponding values in matrixB
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;
            if (aRows != bRows || aCols != bCols)
                throw new Exception("Non-conformable matrices in MatrixAreEqual");

            for (int i = 0; i < aRows; ++i) // each row of A and B
                for (int j = 0; j < aCols; ++j) // each col of A and B
                                                //if (matrixA[i][j] != matrixB[i][j])
                    if (Math.Abs(matrixA[i][j] - matrixB[i][j]) > epsilon)
                        return false;
            return true;
        }
        //=============================================================================================
        static double[][] MatrixProduct(double[][] matrixA, double[][] matrixB)
        {
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;
            if (aCols != bRows)
                throw new Exception("Non-conformable matrices in MatrixProduct");

            double[][] result = MatrixCreate(aRows, bCols);

            for (int i = 0; i < aRows; ++i) // each row of A
                for (int j = 0; j < bCols; ++j) // each col of B
                    for (int k = 0; k < aCols; ++k) // could use k less-than bRows
                        result[i][j] += matrixA[i][k] * matrixB[k][j];

            return result;
        }

        //=========================================================================================
        static double[][] MatrixInverse(double[][] matrix)
        {
            int n = matrix.Length;
            double[][] result = MatrixDuplicate(matrix);

            int[] perm;
            int toggle;
            double[][] lum = MatrixDecompose(matrix, out perm,
              out toggle);
            if (lum == null)
                throw new Exception("Unable to compute inverse");
            double[] b = new double[n];
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == perm[j])
                        b[j] = 1.0;
                    else
                        b[j] = 0.0;
                }
                double[] x = HelperSolve(lum, b);
                for (int j = 0; j < n; ++j)
                    result[j][i] = x[j];
            }
            return result;
        }

        //=========================================================================================
        static double MatrixDeterminant(double[][] matrix)
        {
            int[] perm;
            int toggle;
            double[][] lum = MatrixDecompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new Exception("Unable to compute MatrixDeterminant");
            double result = toggle;
            for (int i = 0; i < lum.Length; ++i)
                result *= lum[i][i];
            return result;
        }
        //==========================================================================================
        static double[][] MatrixDuplicate(double[][] matrix)
        {
            // allocates/creates a duplicate of a matrix.
            double[][] result = MatrixCreate(matrix.Length, matrix[0].Length);
            for (int i = 0; i < matrix.Length; ++i) // copy the values
                for (int j = 0; j < matrix[i].Length; ++j)
                    result[i][j] = matrix[i][j];
            return result;
        }
        //=========================================================================================== 
        static double[] HelperSolve(double[][] luMatrix, double[] b)
        {
            // before calling this helper, permute b using the perm array
            // from MatrixDecompose that generated luMatrix
            int n = luMatrix.Length;
            double[] x = new double[n];
            b.CopyTo(x, 0);
            for (int i = 1; i < n; ++i)
            {
                double sum = x[i];
                for (int j = 0; j < i; ++j)
                    sum -= luMatrix[i][j] * x[j];
                x[i] = sum;
            }
            x[n - 1] /= luMatrix[n - 1][n - 1];
            for (int i = n - 2; i >= 0; --i)
            {
                double sum = x[i];
                for (int j = i + 1; j < n; ++j)
                    sum -= luMatrix[i][j] * x[j];
                x[i] = sum / luMatrix[i][i];
            }
            return x;
        }

        //========================================================================================
        static double[] SystemSolve(double[][] A, double[] b)
        {
            // Solve Ax = b
            int n = A.Length;
            // 1. decompose A
            int[] perm;
            int toggle;
            double[][] luMatrix = MatrixDecompose(A, out perm, out toggle);
            if (luMatrix == null)
                return null;
            // 2. permute b according to perm[] into bp
            double[] bp = new double[b.Length];
            for (int i = 0; i < n; ++i)
                bp[i] = b[perm[i]];
            // 3. call helper
            double[] x = HelperSolve(luMatrix, bp);
            return x;
        }

        // SystemSolve
        //==================================================================================
        static string VectorAsString(double[] vector)
        {
            string s = "";
            for (int i = 0; i < vector.Length; ++i)
                s += vector[i].ToString("F3").PadLeft(8) + Environment.NewLine;
            s += Environment.NewLine;
            return s;
        }
        //=====================================================================================
        static string VectorAsString(int[] vector)
        {
            string s = "";
            for (int i = 0; i < vector.Length; ++i)
                s += vector[i].ToString().PadLeft(2) + " ";
            s += Environment.NewLine;
            return s;
        }

        //===========================================================================================
        static double[][] MatrixDecompose(double[][] matrix, out int[] perm, out int toggle)
        {
            // Doolittle LUP decomposition with partial pivoting.
            // rerturns: result is L (with 1s on diagonal) and U;
            // perm holds row permutations; toggle is +1 or -1 (even or odd)
            int rows = matrix.Length;
            int cols = matrix[0].Length; // assume square
            if (rows != cols)
                throw new Exception("Attempt to decompose a non-square m");
            int n = rows; // convenience
            double[][] result = MatrixDuplicate(matrix);
            perm = new int[n]; // set up row permutation result
            for (int i = 0; i < n; ++i) { perm[i] = i; }
            toggle = 1; // toggle tracks row swaps.
                        // +1 -greater-than even, -1 -greater-than odd. used by MatrixDeterminant
            for (int j = 0; j < n - 1; ++j) // each column
            {
                double colMax = Math.Abs(result[j][j]); // find largest val in col
                int pRow = j;
                // reader Matt V needed this:
                for (int i = j + 1; i < n; ++i)
                {
                    if (Math.Abs(result[i][j]) > colMax)
                    {
                        colMax = Math.Abs(result[i][j]);
                        pRow = i;
                    }
                }
                // Not sure if this approach is needed always, or not.

                if (pRow != j) // if largest value not on pivot, swap rows
                {
                    double[] rowPtr = result[pRow];
                    result[pRow] = result[j];
                    result[j] = rowPtr;

                    int tmp = perm[pRow]; // and swap perm info
                    perm[pRow] = perm[j];
                    perm[j] = tmp;

                    toggle = -toggle; // adjust the row-swap toggle
                }
                // --------------------------------------------------
                // This part added later (not in original)
                // and replaces the 'return null' below.
                // if there is a 0 on the diagonal, find a good row
                // from i = j+1 down that doesn't have
                // a 0 in column j, and swap that good row with row j
                // --------------------------------------------------

                if (result[j][j] == 0.0)
                {
                    // find a good row to swap
                    int goodRow = -1;
                    for (int row = j + 1; row < n; ++row)
                    {
                        if (result[row][j] != 0.0)
                            goodRow = row;
                    }
                    if (goodRow == -1)
                        throw new Exception("Cannot use Doolittle's method");
                    // swap rows so 0.0 no longer on diagonal
                    double[] rowPtr = result[goodRow];
                    result[goodRow] = result[j];
                    result[j] = rowPtr;

                    int tmp = perm[goodRow]; // and swap perm info
                    perm[goodRow] = perm[j];
                    perm[j] = tmp;

                    toggle = -toggle; // adjust the row-swap toggle
                }
                // --------------------------------------------------
                // if diagonal after swap is zero . .
                //if (Math.Abs(result[j][j]) less-than 1.0E-20) 
                //  return null; // consider a throw
                for (int i = j + 1; i < n; ++i)
                {
                    result[i][j] /= result[j][j];
                    for (int k = j + 1; k < n; ++k)
                    {
                        result[i][k] -= result[i][j] * result[j][k];
                    }
                }
            } // main j column loop
            return result;
        }

    }
}
