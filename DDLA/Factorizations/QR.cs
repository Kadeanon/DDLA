// These algorithms are ported from LibFlame.
// https://github.com/flame/libflame

using DDLA.BLAS;
using DDLA.Core;
using DDLA.Misc;
using DDLA.Misc.Flags;

using static DDLA.BLAS.BlasProvider;
using static DDLA.Transformations.HouseHolder;

namespace DDLA.Factorizations;

public class QR
{
    internal static int BlockSize = 256;

    readonly MatrixView matrix;
    readonly MatrixView aux;
    bool computed;
    bool deconstructed;

    public QR(MatrixView mat, bool inplace = false)
    {
        if (inplace)
            matrix = mat;
        else
            matrix = mat.Clone();

        aux = CreateT(matrix);
        computed = false;
    }

    public void ComputeOnce()
    {
        if (deconstructed)
            throw new InvalidOperationException(
                "Matrix has been deconstructed, cannot compute again.");
        if (computed)
            return;
        computed = true;
        QRDecompose(matrix, aux);
    }

    public Vector Solve(VectorView B, VectorView? output = null)
    {
        ComputeOnce();
        if (output is VectorView X)
        {
            if (X.Length != matrix.Cols)
                throw new ArgumentException(
                    "Output matrix has incorrect dimensions.");
        }
        else
        {
            X = Vector.Create(matrix.Cols);
        }
        Solve(matrix, aux, new(B), new(X));
        return new(X);
    }

    public Matrix Solve(MatrixView B, MatrixView? output = null)
    {
        ComputeOnce(); 
        if(output is MatrixView X)
        {
            if (X.Rows != matrix.Cols || X.Cols != B.Cols)
                throw new ArgumentException(
                    "Output matrix has incorrect dimensions.");
        }
        else
        {
            X = Matrix.Create(matrix.Cols, B.Cols);
        }
        Solve(matrix, aux, B, X);
        return new(X);
    }

    public Vector ApplyQ(VectorView B)
    {
        var W = Matrix.Create(aux.Rows, 1);
        var Y = new MatrixView(B);
        ApplyQlnfc(matrix, aux, W, Y);
        return new(B);
    }

    public Matrix ApplyQ(MatrixView B)
    {
        var W = Matrix.Create(aux.Rows, B.MinDim);
        var Y = B;
        ApplyQlnfc(matrix, aux, W, Y);
        return new(B);
    }

    public Vector ApplyQT(VectorView B)
    {
        var W = Matrix.Create(aux.Rows, 1);
        var Y = new MatrixView(B);
        ApplyQlhfc(matrix, aux, W, Y);
        return new(B);
    }

    public Matrix ApplyQT(MatrixView B)
    {
        var W = Matrix.Create(aux.Rows, B.MinDim);
        var Y = B;
        ApplyQlhfc(matrix, aux, W, Y);
        return new(B);
    }

    public void Deconstruct(out Matrix Q, out Matrix R)
    {
        ComputeOnce();

        Q = Matrix.Eyes(matrix.Rows);
        FormQ(matrix, aux, Q);

        R = new(matrix);
        Set(DiagType.Unit, UpLo.Lower, 0.0, matrix);

        deconstructed = true;
    }

    public static Matrix CreateT(MatrixView A)
    {
        int block = Math.Min(BlockSize, A.MaxDim);
        return Matrix.Create(block, A.Cols);
    }

    public static void QRDecompose(MatrixView A, out MatrixView T)
    {
        T = CreateT(A);
        if (A.Cols > BlockSize)
            QRDecBlock(A, T);
        else
            QRDecUnblock(A, T);
    }

    public static void QRDecompose(MatrixView A, MatrixView T)
    {
        int n = A.Cols;
        if (T.Cols != n)
            throw new ArgumentException(
                "Matrix T and A must have the same number of columns.");
        if (n > BlockSize)
            QRDecBlock(A, T);
        else
            QRDecUnblock(A, T);
    }

    public static void Solve(MatrixView A, MatrixView T,
        MatrixView B, MatrixView X)
    {
        if (A.Cols != T.Cols)
            throw new ArgumentException(
                "Matrix T and A must have the same number of columns.");
        if (A.Rows != B.Rows)
            throw new ArgumentException(
                "Matrix A and B must have the same number of rows.");
        if (A.Cols != X.Rows)
            throw new ArgumentException(
                "The number of columns in A must match the number of rows in X.");
        if (X.Cols != B.Cols)
            throw new ArgumentException(
                "Matrix X and B must have the same number of columns.");

        SolveUncheck(A, T, B, X);
    }

    public static void FormQ(MatrixView A,
        MatrixView T)
    {
        if (A.Cols != T.Cols)
            throw new ArgumentException(
                "Matrix T and A must have the same number of columns.");
        FormQUncheck(A, T);
    }

    public static void FormQ(MatrixView A,
        MatrixView T, MatrixView Q)
    {
        if (A.Cols != T.Cols)
            throw new ArgumentException(
                "Matrix T and A must have the same number of columns.");
        if (A.Rows != Q.Rows || A.Rows != Q.Cols)
            throw new ArgumentException(
                "The dimension of Q must be equal to the number of rows in A.");

        FormQUncheck(A, T, Q);
    }

    internal static void FormQUncheck(MatrixView A,
        MatrixView T, MatrixView Q)
    {
        Q.Fill(0);
        Q.Diag.Fill(1);

        // Q = H_{0} H_{1} ... H_{k-1}
        var W = Matrix.Create(T.Rows, Q.MaxDim);
        ApplyQlnfc(A, T, W, Q);
    }

    internal static void FormQUncheck(MatrixView A,
        MatrixView T)
    {
        A.MakeTr(UpLo.Lower);

        var slice = ..A.Cols;
        if (T.Cols > A.Cols)
            T = T[.., slice];

        slice = T.Cols..;
        if (A.Cols > T.Cols)
        {
            var QBR = A[slice, slice];
            QBR.MakeTr(UpLo.Upper);
            QBR.Diag.Fill(0);
        }

        // Set the digaonal to one.
        A.Diag.Fill(1);

        var W = Matrix.Create(T.Rows, A.MaxDim);

        FormQBlock(A, T, W);

    }

    internal static void FormQUnblock(MatrixView A, MatrixView T)
    {
        int min_m_n = A.MinDim;
        int i;

        for (i = min_m_n - 1; i >= 0; --i)
        {
            var slice = (i + 1)..;
            ref var alphA11 = ref A[i, i];
            var a21 = A[slice, i];
            var a12t = A[i, slice];
            var A22 = A[slice, slice];

            ref double tau11 = ref T[i, i];

            ApplyHouseHolder(SideType.Left,
                tau11, a21, a12t, A22);

            alphA11 = 1 - 1 / tau11;

            InvScal(-tau11, a21);
        }
    }

    internal static void FormQBlock(MatrixView A, MatrixView T, MatrixView W)
    {
        int block, b_alg;
        int m_BR, n_BR;

        b_alg = T.Rows;


        // If A is wider than T, then we need to position ourseves carefully
        // within the matrix for the initial partitioning.
        if (A.Cols > T.Cols)
        {
            m_BR = A.Rows - T.Cols;
            n_BR = A.Cols - T.Cols;
        }
        else
        {
            m_BR = A.Rows - A.Cols;
            n_BR = 0;
        }
        var partA = PartitionGrid.Create
            (A, m_BR, n_BR, Quadrant.BottomRight,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Right,
            out var t0, out var t1, out var t2);

        while (t0.Cols > 0)
        {
            block = Math.Min(b_alg, A00.MinDim);

            if (t2.Cols == 0 && T.Cols % b_alg > 0)
                block = T.Cols % b_alg;

            var abr = A22;
            using var partAStep = partA.Step(block, block);
            using var partTStep = partT.Step(block);

            var t1t = t1.SliceSubUncheck(0, block, 0, t1.Cols);
            var WTL = W.SliceSubUncheck(0, block, 0, A12.Cols);

            if (abr.Rows == 0)
                FormQUnblock(A11, t1t);
            else
            {
                PartUtils.Merge21to11
                    (A11,
                     A21, out var ab1);
                PartUtils.Merge21to11
                    (A12,
                     A22, out var ab2);

                ApplyQlnfc(ab1, t1t, WTL, ab2);
                FormQUnblock(ab1, t1t);
            }
        }
    }

    internal static void QRDecUnblock(MatrixView A, MatrixView T)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);

        var partT = PartitionGrid.Create
            (T, 0, 0, Quadrant.TopLeft,
            out var T00, out var t01, out var T02,
            out var t10, out var t11, out var t12,
            out var T20, out var t21, out var T22);

        while (A22.MinDim > 0)
        {
            using var partAStep = partA.Step();
            using var partTStep = partT.Step();

            ref var alpha = ref a11.GetHeadRef();
            ref var tau = ref t11.GetHeadRef();
            var a21v = a21.GetColumn(0);
            var a12t = a12.GetRow(0);

            BuildHouseHolder(ref alpha, a21v, out tau);

            ApplyHouseHolder(SideType.Left,
                tau, a21v, a12t, A22);

            var a10t = a10.GetRow(0);
            var t01v = t01.GetColumn(0);
            // T01 = A10t' + A20' * A21;
            a10t.CopyTo(t01v);
            GeMV(
                1, A20.T, a21v, 1, t01v);
        }
    }

    internal static void QRDecBlock(MatrixView A, MatrixView T)
    {
        // Query the algorithmic blocksize by inspecting the length of T.
        int blockside = T.Rows;
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Left,
            out var T0, out var T1, out var T2);

        while (A22.MinDim > 0)
        {
            int b = Math.Min(blockside, A22.MinDim);

            using var partAStep = partA.Step(b, b);
            using var partTStep = partT.Step(b);

            var T1t = T1.SliceSubUncheck(0, b, 0, T1.Cols);
            PartUtils.Merge21to11(A11,
                                 A21, out var Ab1);

            QRDecUnblock(Ab1, T1t);


            if (A12.Cols > 0)
            {
                PartUtils.Merge21to11(A12,
                                      A22, out var Ab2);

                ApplyQlhfc(Ab1, T1t, T2, Ab2);
            }

        }
    }

    internal static void SolveUncheck(MatrixView A, MatrixView T,
        MatrixView B, MatrixView X)
    {
        var W = Matrix.Create(T.Rows, B.Cols);
        var Y = B.Clone().View;
        ApplyQlhfc(A, T, W, Y);
        Range widRange = ..(A.Cols);
        var AT = A[widRange, ..];
        var YT = Y[widRange, ..];
        TrSM(SideType.Left, UpLo.Upper,
            1, AT, YT);
        YT.CopyTo(X);
    }

    /// <summary>
    /// Apply a transpose of a unitary matrix Q to matrix B from the left: B := Q' B, 
    /// where Q is the forward product of Householder transformations,
    /// and H(i) corresponds to the householder vector stored below the diagonal
    /// in the ith column of A.
    /// </summary>
    /// <param name="A">Matrix A, which storage the householder vectors
    /// in its strictly lower triangle.</param>
    /// <param name="T">Matrix T, which storage the triangular factors
    /// in its strictly upper triangle.</param>
    /// <param name="W">The workspace matrix.</param>
    /// <param name="B">The target matrix B.</param>
    public static void ApplyQlhfc(MatrixView A, MatrixView T, MatrixView W, MatrixView B)
    {

        int block = T.Rows;
        int width = B.Cols;
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);
        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Left,
            out var T0, out var T1, out var T2);
        var partB = PartitionVertical.Create
            (B, 0, UpLo.Upper,
            out var B0, out var B1, out var B2);

        while (A22.MinDim > 0)
        {

            int b = Math.Min(block, A22.MinDim);
            using var partAStep = partA.Step(b, b);
            using var partTStep = partT.Step(b);
            using var partBStep = partB.Step(b);

            var slice = ..(b);
            var T1t = T1[slice, ..];
            var wtl = W[slice, ..];

            B1.CopyTo(wtl);

            TrMM(SideType.Left, UpLo.Lower,
                TransType.OnlyTrans, DiagType.Unit,
                1, A11, wtl);

            GeMM(1, A21.T,
                B2, 1, wtl);

            TrSM(SideType.Left, UpLo.Upper,
                TransType.OnlyTrans, DiagType.NonUnit,
                1, T1t, wtl);

            GeMM(-1, A21, wtl, 1, B2);

            TrMM(SideType.Left, UpLo.Lower,
                TransType.NoTrans, DiagType.Unit,
                -1, A11, wtl);

            Axpy(1, wtl, B1);
        }
    }

    public static void ApplyQlnfc(MatrixView A, MatrixView T, MatrixView W, MatrixView B)
    {
        int block = T.Rows;
        int b;

        int m_BR = 0;
        int n_BR = 0;
        if (A.Rows > A.Cols)
            m_BR = A.Rows - A.Cols;
        else if (A.Rows < A.Cols)
            n_BR = A.Cols - A.Rows;

        var partA = PartitionGrid.Create
            (A, m_BR, n_BR, Quadrant.BottomRight,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        var partT = PartitionHorizontal.Create(
            T, A.MinDim, SideType.Left,
            out var T0, out var T1, out var T2,
            backward: true);

        var partB = PartitionVertical.Create(
            B, m_BR, UpLo.Lower,
        out var B0, out var B1, out var B2);

        while (A00.MinDim > 0)
        {
            b = Math.Min(block, A00.MinDim);

            if (T2.Cols == 0 && T.Cols % block > 0)
                b = T.Cols % block;

            using var partAStep = partA.Step(b, b);
            using var partTStep = partT.Step(b);
            using var partBStep = partB.Step(b);

            PartUtils.Part11to21
                (T1, b, UpLo.Upper, out var T1t,
                                    out var T2b);

            PartUtils.Part11to22
                (W, b, B1.Cols, Quadrant.TopLeft,
                out var w00, out var w01,
                out var w10, out var w11);

            B1.CopyTo(w00);
            TrMM(SideType.Left, UpLo.Lower,
                TransType.OnlyTrans, DiagType.Unit,
                1, A11, w00);
            GeMM(1, A21.T,
                B2, 1, w00);
            TrSM(SideType.Left, UpLo.Upper,
                TransType.NoTrans, DiagType.NonUnit,
               1, T1t, w00);

            GeMM(-1, A21, w00, 1, B2);
            TrMM(SideType.Left, UpLo.Lower,
                TransType.NoTrans, DiagType.Unit,
                -1, A11, w00);
            Axpy(1, w00, B1);
        }
    }

}
