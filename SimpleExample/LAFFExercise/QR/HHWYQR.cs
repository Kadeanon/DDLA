using DDLA.BLAS;
using DDLA.Misc.Flags;
using System.Diagnostics;

namespace SimpleExample.LAFFExercise.QR;

/// <summary>
/// Blocked Householder QR factorization
/// with WY representation
/// </summary>
public class HHWYQR(Matrix A) : QRBase(A)
{
    public override bool IsEconomy => true;

    public const int MaxBlockSize = 4;

    public override void Kernel()
    {
        int m = this.A.Rows;
        int n = this.A.Cols;
        this.R = Matrix.Create(n, n);
        Q = this.A;

        var R = this.R.View;
        var A = this.A.View;

        var blksz = Math.Min(MaxBlockSize, n);
        var S = Matrix.Create(m, blksz).View;
        var Work = Matrix.Create(m, blksz).View;
        for(int i = 0; i < n; i += blksz)
        {
            var block = Math.Min(blksz, n - i);
            var A1 = A[i.., i..(i + block)];
            var A2 = A[i.., (i + block)..];

            var S1 = S[i..(i + block), ..block];
            UnblockHH(A1, S1);
            UpdateRight(A1, S1, A2, Work);
        }
        for (int j = 0; j < n; j++)
        {
            var r01 = R[..j, j];
            ref var r11 = ref R[j, j];
            var u01 = A[..j, j]; 
            ref var u11 = ref A[j, j];
            u01.CopyTo(r01);
            r11 = u11;
            u01.Fill(0);
            u11 = 1.0;
        }

        throw new NotImplementedException();
        var W = S.Multify(A.T);
        BlasProvider.TrMM(SideType.Right, UpLo.Upper, -1.0, W, A);
        A.ShiftDiag(1.0);
    }

    private static void UnblockHH(MatrixView A, MatrixView S)
    {
        int n = A.Cols;
        for (int j = 0; j < n; j++)
        {
            // Build HH to zero out entries below the diagonal in column j

            var S00 = S[..j, ..j];
            var s01 = S[..j, j];
            ref var s11 = ref S[j, j];

            var u10 = A[j, ..j];
            ref var u11 = ref A[j, j];
            var U20 = A[(j + 1).., ..j];
            var u21 = A[(j + 1).., j];

            var xSq = u21.SumSq();
            var alphaSq = u11 * u11 + xSq;
            var alpha = Math.Sqrt(alphaSq);
            var rho = -Math.Sign(u11) * alpha;
            var miu = 1 / (u11 - rho);
            u11 = rho;
            u21.Scaled(miu);
            s11 = 2 / (1 + xSq * miu * miu);
            u21.LeftMul(U20, output: s01);
            s01.Added(u10);
            BlasProvider.TrMV(UpLo.Upper, -s11, S00, s01);


            // Apply HH to remaining columns
            for (var col = j + 1; col < n; col++)
            {
                ref var a12 = ref A[j, col];
                var a22 = A[(j + 1).., col];

                var w = a12 + u21 * a22;
                w *= s11;
                a12 -= w;
                a22.Added(-w, u21);
            }
        }
    }

    private static void UpdateRight(MatrixView U0, MatrixView S, MatrixView U1, MatrixView Work)
    {
        var n1 = U1.Cols;
        if (n1 == 0) return;
        var n = U0.Cols;

        var block = Math.Min(n, n1);
        for (int j = 0; j < n1; j += block)
        {
            block = Math.Min(block, n1 - j);
            UpdateRightBlock(U0, S, U1[.., j..(j + block)], Work[..n, ..block]);
        }
    }

    private static void UpdateRightBlock(MatrixView U0, MatrixView S, MatrixView U1, MatrixView Work)
    {
        var m = U0.Rows;
        var n = U0.Cols;
        var n1 = U1.Cols;

        var U00 = U0[..n, ..];
        var U10 = U0[n.., ..];
        var U01 = U1[..n, ..];
        var U11 = U1[n.., ..];

        //Work = U01
        U01.CopyTo(Work);
        //Work = (tril(U10))^T * U00
        BlasProvider.TrMM(SideType.Left, UpLo.Lower,
            TransType.OnlyTrans, DiagType.Unit,
            1.0, U00, Work);
        //Work += U11^T * U10
        BlasProvider.GeMM(TransType.OnlyTrans, TransType.NoTrans,
            1.0, U10, U11, 1.0, Work);
        //Work = S * Work
        BlasProvider.TrMM(SideType.Left, UpLo.Upper,
            TransType.NoTrans, DiagType.NonUnit,
            1.0, S, Work);
        //U01 -= U00 * Work
        BlasProvider.GeMM(-1.0, U00, Work, 1.0, U01);
        //U11 -= U10 * Work
        BlasProvider.GeMM(-1.0, U10, Work, 1.0, U11);
    }
}
