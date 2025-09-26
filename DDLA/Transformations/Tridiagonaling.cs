using DDLA.BLAS;
using DDLA.Core;
using DDLA.Factorizations;
using DDLA.Misc;
using DDLA.Misc.Flags;
using DDLA.Misc.Pools;
using static DDLA.BLAS.BlasProvider;

namespace DDLA.Transformations;

public static class Tridiagonaling
{
    public static int BlockSize { get; } = 64;

    public static Matrix CreateT(MatrixView A)
    {
        return Matrix.Create
            (Math.Min(BlockSize, A.Rows), A.Rows);
    }

    /// <summary>
    /// Reduce a symmetric matrix A to tridiagonal form using
    /// blocked HouseHolder UT Transformation. 
    /// The method only use the lower triangular part of A.
    /// </summary>
    /// <param name="A">Orig matrix A, when back, the lower
    /// triangular part of A will be overwritten by the 
    /// HouseHolder vectors.</param>
    /// <param name="T">A workspace matrix to save the upper 
    /// triangular factors of the block Householder transformations</param>
    /// <param name="d">A vector to save the diag elements.</param>
    /// <param name="e">A vector to save the subdiag elements.</param>
    /// <remarks>After called this method, you can call 
    /// <see cref="FormQ(MatrixView, MatrixView)"/> to get the
    /// Q matrix overwritten in full A matrix.</remarks>
    public static void Tridiag(MatrixView A, MatrixView T,
        VectorView d, VectorView e)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);
        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Left,
            out var T0, out var T1, out var T2);
        int index = 0;
        while (A22.Rows > 0)
        {
            int block = Math.Min(T.Rows, A22.Rows);
            var indexNext = index + block;
            var ABR = A22;
            using var partAStep = partA.Step(block, block);
            using var partTStep = partT.Step(block);

            var subDiagEnd = Math.Min(indexNext, e.Length);
            var d1 = d[index..indexNext];
            var e1 = e[index..subDiagEnd];
            Step(ABR, T1[..block, ..], d1, e1);
            index = indexNext;
        }
        FormQ(A, T);
    }

    /// <summary>
    /// Reduce a symmetric matrix A to tridiagonal form using
    /// blocked HouseHolder UT Transformation. 
    /// The method only use the lower triangular part of A.
    /// </summary>
    /// <param name="A">Orig matrix A, when back, the lower
    /// triangular part of A will be overwritten by the 
    /// HouseHolder vectors.</param>
    /// <param name="T">A workspace matrix to save the upper 
    /// triangular factors of the block Householder transformations</param>
    /// <param name="d">A vector to save the diag elements.</param>
    /// <param name="e">A vector to save the subdiag elements.</param>
    /// <remarks>After called this method, you can call 
    /// <see cref="FormQ(MatrixView, MatrixView)"/> to get the
    /// Q matrix overwritten in full A matrix.</remarks>
    public static void Tridiag(MatrixView A, out MatrixView T,
        out VectorView d, out VectorView e)
    {
        T = CreateT(A);
        d = Vector.Create(A.Rows);
        e = Vector.Create(A.Rows - 1);
        Tridiag(A, T, d, e);
    }

    private static void Step(MatrixView A, MatrixView T,
        VectorView d, VectorView e)
    {
        int ARows = A.Rows;
        int TRows = T.Rows;

        using var tmpHandle = 
            InternelPool.TakeVector(ARows, out var tmp);

        for (var i = 0; i < TRows; i++)
        {
            d[i] = A[i, i];
            if (i < ARows - 1)
            {
                var A20 = A[(i + 1).., ..i];
                var a21 = A[(i + 1).., i];
                var A22 = A[(i + 1).., (i + 1)..];
                var t01 = T[..i, i];
                ref double tau11 = ref T[i, i];
                var p = tmp[(i + 1)..];

                BuildHH(a21, out var sigma, out tau11);

                SyMV(UpLo.Lower, 1, A22, a21, 0, p);
                var beta = a21 * p / (2 * tau11);
                p.AddedBy(-beta, a21);
                p.InvScaled(tau11);
                A22.Rank2(UpLo.Lower,
                    -1.0, a21, p);

                a21.LeftMul(A20, t01);
                a21.GetHeadRef() = sigma;
                e[i] = sigma;
            }
        }
    }

    /// <summary>
    /// Generate the full orthogonal matrix Q from the matrices A and T
    /// after the Tridiag method is completed. The resulting Q matrix
    /// will directly overwrite the entire A matrix.
    /// </summary>
    /// <param name="A">The matrix A, which will be overwritten by the
    /// resulting orthogonal matrix Q.</param>
    /// <param name="T">The workspace matrix T containing the upper
    /// triangular factors of the block Householder transformations.</param>
    public static void FormQ(MatrixView A, MatrixView T)
    {
        for (int j = A.Rows - 2; j > 0; --j)
        {
            A[(j + 1).., j - 1].CopyTo(A[(j + 1).., j]);
        }
        A[0, ..].Fill(0);
        A[.., 0].Fill(0);
        A.Diag.Fill(1);

        A = A[1.., 1..];
        T = T[.., ..^1];
        A.MakeTr(UpLo.Lower);

        var slice = ..A.Cols;
        if (T.Cols > A.Cols)
            T = T[.., slice];

        var W = Matrix.Create(T.Rows, A.Cols);

        int block, blockSize;

        blockSize = T.Rows;

        var partA = PartitionGrid.Create
            (A, A.Rows - A.Cols, 0, Quadrant.BottomRight,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Right,
            out var T0, out var T1, out var T2);

        while (T0.Cols > 0)
        {
            block = Math.Min(blockSize, A00.MinDim);

            if (T2.Cols == 0 && T.Cols % blockSize > 0)
                block = T.Cols % blockSize;

            var ABR = A22;
            using var partAStep = partA.Step(block, block);
            using var partTStep = partT.Step(block);

            var TT1 = T1[..block, ..];
            var WTL = W[..block, ..A12.Cols];
            var AB1 = A11;

            if (ABR.Rows != 0)
            {
                PartUtils.Merge21to11
                    (A11,
                     A21, out AB1);
                PartUtils.Merge21to11
                    (A12,
                     A22, out var AB2);

                QR.ApplyQlnfc(AB1, TT1, WTL, AB2);
            }

            for (var i = AB1.MinDim - 1; i >= 0; --i)
            {
                ref var alphA11 = ref AB1[i, i];
                var a21 = AB1[(i + 1).., i];

                ref double tau11 = ref TT1[i, i];

                HouseHolder.ApplyHouseHolder(SideType.Left,
                    ref tau11, a21,
                    AB1[i.., (i + 1)..]);

                alphA11 = 1 - 1 / tau11;

                a21.InvScaled(-tau11);
            }
        }
    }

    private static void BuildHH(VectorView x, out double sigma, out double tau)
    {
        if (x.Length == 0)
            throw new ArgumentException("Vector length must be at least 1.", nameof(x));

        ref var chi = ref x[0];
        var xLast = x[1..];

        double lenLast = xLast.NrmF();

        if (lenLast == 0.0)
        {
            sigma = -chi;
            chi = 1;
            tau = 0.5;
            return;
        }

        double lenx = double.Hypot(chi, lenLast);
        sigma = double.CopySign(lenx, chi);
        double invScale = 1 / (chi - sigma);
        chi = 1;
        xLast.Scaled(invScale);
        lenLast *= invScale;
        tau = (1 + lenLast * lenLast) / 2;
    }
}
