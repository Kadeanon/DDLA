using DDLA.BLAS;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleExample.LAFFExercise.QR
{
    /// <summary>
    /// Unblocked Householder QR
    /// </summary>
    public class HHQR(Matrix A)
        : QRBase(A)
    {
        public override bool IsEconomy => true;

        public override void Kernel()
        {
            int m = A.Rows;
            int n = A.Cols;
            this.R = Matrix.Create(n, n);
            Q = A;

            var R = this.R.View;

            var taus = new double[n];
            for (int j = 0; j < n; j++)
            {
                ref var tau = ref taus[j];
                // Build HH to zero out entries below the diagonal in column j
                BuildHH(A, j, j, out tau);
                // Apply HH to remaining columns
                ApplyHH(A, j, j, j + 1, n, tau);
            }
            // Extract R from modified A
            // TODO: Use Copy with Uplo
            for (int j = 0; j < n; j++)
            {
                A[..(j + 1), j].CopyTo(R[..(j + 1), j]);
            }

            // Form Q overwrite A
            FormQ(A, taus);
        }

        private static void BuildHH(MatrixView A, int i, int j,
            out double tau)
        {
            double a11 = A[i, j];
            var subVec = A[(i + 1).., j];
            double xSq = subVec.SumSq();
            var m = A.Rows;
            double alphaSq = a11 * a11 + xSq;
            double alpha = Math.Sqrt(alphaSq);
            double rho = -Math.Sign(a11) * alpha;
            double miu = a11 - rho;
            tau = 1;
            for (int k = i + 1; k < m; k++)
            {
                var akj = A[k, j];
                akj /= miu;
                A[k, j] = akj;
                tau += akj * akj;
            }
            A[i, j] = rho;
            tau /= 2;
        }

        private static void ApplyHH(MatrixView A, int i, int j,
            int colStart, int colEnd, double tau)
        {
            var m = A.Rows;
            for (var col = colStart; col < colEnd; col++)
            {
                var a12 = A[i, col];
                var w = a12;
                for (int k = i + 1; k < m; k++)
                {
                    var ak1 = A[k, j];
                    var ak2 = A[k, col];
                    w += ak1 * ak2;
                }
                w /= tau;
                A[i, col] = a12 - w;
                for (int k = i + 1; k < m; k++)
                {
                    var ak1 = A[k, j];
                    var ak2 = A[k, col];
                    A[k, col] = ak2 - ak1 * w;
                }
            }
        }

        private static void FormQ(MatrixView A, double[] taus)
        {
            var m = A.Rows;
            var n = A.Cols;
            for (int j = n - 1; j >= 0; j--)
            {
                var negInvTau = -1 / taus[j];
                // Apply HH to A as Q
                A[j, j] = 1.0 + negInvTau;

                for (int col = j + 1; col < n; col++)
                {
                    var w = 0.0;
                    for (int row = j + 1; row < m; row++)
                    {
                        w += A[row, j] * A[row, col];
                    }
                    w *= negInvTau;
                    A[j, col] = w;
                    for (int row = j + 1; row < m; row++)
                    {
                        A[row, col] += A[row, j] * w;
                    }
                }

                for (int i = j + 1; i < m; i++)
                {
                    A[i, j] *= negInvTau;
                }
            }
        }
    }
}
