namespace SimpleExample.LAFFExercise.QR;

public class MGSQR(Matrix A)
    : QRBase(A)
{
    override public bool IsEconomy => true;

    public override void Kernel()
    {
        Q = this.A;
        var A = this.A.View;
        int m = A.Rows;
        int n = A.Cols;
        this.R = Matrix.Create(n, n);
        var R = this.R.View;

        for (int j = 0; j < n; j++)
        {
            // A = | Am0 amj Am2 |
            // Q = | Qm0 qmj Qm2 |
            // R = | R00 r0j R02 |
            //     |   0 r1j R12 |
            //     |   0 r2j R22 |

            // r1j := ‖a1‖2
            var rho11 = A[..m, j].SumSq();
            rho11 = Math.Sqrt(rho11);
            R[j, j] = rho11;
            var amj = A[.., j];
            var Am2 = A[.., (j + 1)..];
            var r12 = R[j, (j + 1)..];


            // amj := amj/ρ11
            amj.InvScaled(rho11);

            // R12 := Amj^H * Am2
            amj.LeftMul(Am2, output: r12);

            // Am2 -= amj * R12
            Am2.Rank1(-1.0, amj, r12);
        }
    }
}
