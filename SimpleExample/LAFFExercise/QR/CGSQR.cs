using DDLA.BLAS;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleExample.LAFFExercise.QR
{
    /// <summary>
    /// Classic Gram-Schmidt QR factorization
    /// </summary>
    public class CGSQR(Matrix A) : QRBase(A)
    {
        public override bool IsEconomy => true;

        public override void Kernel()
        {
            int m = A.Rows;
            int n = A.Cols;
            this.Q = Matrix.Create(m, n);
            this.R = Matrix.Create(n, n);

            var Q = this.Q.View;
            var R = this.R.View;

            for (int j = 0; j < n; j++)
            {
                // A = | Am0 amj Am2 |
                // Q = | Qm0 qmj Qm2 |
                // R = | R00 r0j R02 |
                //     |   0 r1j R12 |
                //     |   0 r2j R22 |

                //r0j := Qm0^H*amj
                for (int i = 0; i < j; i++)
                {
                    R[i, j] = Q[..j, i] * A[..j, j];
                }

                //amj := amj − Qm0*r0j
                //r1j := ‖aj‖2
                var p11 = 0.0;
                for (int i = 0; i < m; i++)
                {
                    // m = [0, m) => i
                    //aij -= Qi0*r0j
                    var aij = A[i, j];
                    aij -= Q[i, ..j] * R[..j, j];
                    //r1j := ‖aij‖2
                    p11 += aij * aij;
                    A[i, j] = aij;
                }
                p11 = Math.Sqrt(p11);
                R[j, j] = p11;

                //qmj := amj/ρ11
                BlasProvider.Scal2(1 / p11, A[.., j], Q[.., j]);
            }
        }
    }
}
