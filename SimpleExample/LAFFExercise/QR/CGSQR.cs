namespace SimpleExample.LAFFExercise.QR;

/// <summary>
/// Classic Gram-Schmidt QR factorization
/// </summary>
public class CGSQR(Matrix A) : QRBase(A)
{
    public override bool IsEconomy => true;

    public override void Kernel()
    {
        int m = this.A.Rows;
        int n = this.A.Cols;
        this.Q = Matrix.Create(m, n);
        this.R = Matrix.Create(n, n);

        var Q = this.Q.View;
        var R = this.R.View;
        var A = this.A.View;

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
            var r1j = 0.0;
            for (int i = 0; i < m; i++)
            {
                // m = [0, m) => i
                //aij -= qi0*r0j
                var aij = A[i, j];
                var qi0 = Q[i, ..j];
                var r0j = R[..j, j];
                aij -= qi0 * r0j;
                //r1j := ‖aij‖2
                r1j += aij * aij;
                A[i, j] = aij;
            }
            r1j = Math.Sqrt(r1j);
            R[j, j] = r1j;

            //qmj := amj/ρ11
            A[.., j].Scale(1 / r1j, Q[.., j]);
        }
    }
}
