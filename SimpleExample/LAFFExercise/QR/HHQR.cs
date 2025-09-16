namespace SimpleExample.LAFFExercise.QR;

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
        ref double a11 = ref A[i, j];
        var A21 = A[(i + 1).., j];
        double xSq = A21.SumSq();
        double alphaSq = a11 * a11 + xSq;
        double alpha = Math.Sqrt(alphaSq);
        double rho = -Math.Sign(a11) * alpha;
        double miu = a11 - rho;
        A21.Scaled(1 / miu);
        tau = 1 + A21.SumSq();
        a11 = rho;
        tau /= 2;
    }

    private static void ApplyHH(MatrixView A, int i, int j,
        int colStart, int colEnd, double tau)
    {
        var a21 = A[(i + 1).., j];

        for (var col = colStart; col < colEnd; col++)
        {
            ref var a12 = ref A[i, col];
            var a22 = A[(i + 1).., col];

            var w = a12 + a21 * a22;
            w /= tau;
            a12 -= w;
            a22.Added(-w, a21);
        }
    }

    private static void FormQ(MatrixView A, double[] taus)
    {
        var n = A.Cols;
        for (int j = n - 1; j >= 0; j--)
        {
            ref var a11 = ref A[j, j];
            var a12 = A[j, (j + 1)..];
            var a21 = A[(j + 1).., j];
            var A22 = A[(j + 1).., (j + 1)..];

            var negInvTau = -1 / taus[j];
            a11 = 1 + negInvTau;
            a21.LeftMul(negInvTau, A22, a12);
            A22.Rank1(a21, a12);
            a21.Scaled(negInvTau);
        }
    }
}
