using DDLA.Core;
using DDLA.Misc;
using DDLA.Misc.Flags;

using static DDLA.BLAS.BlasProvider;

namespace DDLA.Factorizations;

public class LDLT
{
    readonly UpLo uplo;
    readonly MatrixView matrix;
    bool computed;

    public LDLT(UpLo uplo, MatrixView mat, bool inplace = false)
    {
        this.uplo = uplo;
        CheckSymmMatLength(mat, uplo);
        if (inplace)
        {
            matrix = mat;
        }
        else
        {
            matrix = mat.Clone();
        }
        computed = false;
    }

    public void ComputeOnce()
    {
        if (computed)
            return;
        computed = true;
        LDLTDecompose(uplo, matrix);
    }

    /// <summary>
    /// Performs Cholesky decomposition on a symmetric positive-definite matrix A.
    /// The decomposition finds a lower (or upper) triangular matrix L (or U) such that:
    /// <para />
    ///     A = L * L~   (if UpLo == Lower)
    /// <para />
    ///     A = Uᵗ * U   (if UpLo == Upper)
    /// <para />
    /// The result overwrites A with its Cholesky factor.
    /// </summary>
    /// <param name="A">The input matrix (must be square and symmetric positive-definite).</param>
    /// <exception cref="ArgumentException">Thrown if the matrix is not square.</exception>
    /// <exception cref="LinalgException">Thrown if a non-positive pivot is encountered.</exception>

    public static void LDLTDecompose(UpLo uplo, MatrixView A)
    {
        CheckSymmMatLength(A, uplo);
        if (uplo == UpLo.Lower)
        {
            if (A.MinDim > BlockSize)
                LDLTLowerBlock(A);
            else
                LDLTLowerUnblock(A);
        }
        else
        {
            if (A.MinDim > BlockSize)
                LDLTUpperBlock(A);
            else
                LDLTUpperUnblock(A);
        }
    }

    public VectorView Solve(VectorView b, bool inplace = false)
    {
        ComputeOnce();
        var bMat = new MatrixView(b);
        if (!inplace)
            bMat = bMat.Clone();
        LDLTSolve(uplo, matrix, bMat);
        return bMat.GetColumn(0);
    }

    public MatrixView Solve(MatrixView b, bool inplace = false)
    {
        ComputeOnce();
        if (!inplace)
            b = b.Clone();
        LDLTSolve(uplo, matrix, b);
        return b;
    }

    /// <summary>
    /// Solves the linear system A * X = B using the Cholesky factorization of A.
    /// Assumes A has already been factorized or is symmetric positive-definite.
    /// The solution is computed via forward and backward substitution:
    /// <para />
    ///     If A = L * Lᵗ, then solve L * Y = B, then Lᵗ * X = Y.
    /// <para />
    ///     If A = Uᵗ * U, then solve Uᵗ * Y = B, then U * X = Y.
    /// </summary>
    /// <param name="A">The Cholesky factorized matrix.</param>
    /// <param name="B">The right-hand side and solution matrix B. On output, contains the solution X.</param>
    /// <exception cref="ArgumentException">Thrown if matrix dimensions do not match.</exception>
    public static void LDLTSolve(UpLo uplo, MatrixView A,
        MatrixView B)
    {
        var m = CheckSymmMatLength(A, uplo);
        if (B.Rows != m)
            throw new ArgumentException(
                "Matrix dimensions do not match for Cholesky solve.");

        LDLTSolveUncheck(uplo, A, B);
    }

    public static void LDLTSolve(UpLo uplo, MatrixView A,
        MatrixView B, MatrixView X)
    {
        var m = CheckSymmMatLength(A, uplo);
        if (X.Rows != m)
            throw new ArgumentException(
                "Matrix dimensions do not match for Cholesky solve.");
        B.CopyTo(X);

        LDLTSolveUncheck(uplo, A, X);
    }

    public void Deconstruct(out Matrix L, out Matrix D)
    {
        ComputeOnce();

        D = Matrix.Diagonals(matrix.Diag);
        L = new(matrix);
        MakeTr(L, uplo);
        SetDiag(1.0, L);
    }

    internal static int BlockSize = 128;

    internal static void LDLTLowerUnblock(MatrixView A)
    {
        int i = 0;
        ref var a11 = ref A[i, i];
        for (; i < A.Rows - 1; i++)
        {
            a11 = ref A.AtUncheck(i, i);
            if (a11 == 0)
                throw new LinalgException("Cholesky",
                    $"Warning: Cholesky factorization " +
                    $"encountered A non-positive pivot" +
                    $" at index {i} with value {a11}.");
            var a21 = A.SliceColUncheck(i, i + 1);
            var A22 = A.SliceSubUncheck(i + 1, i + 1);
            a21.InvScaled(a11);
            A22.Rank1(UpLo.Lower, -a11, a21);
        }
    }

    internal static void LDLTLowerBlock(MatrixView A)
    {
        MatrixView Work = default;
        if(A.Rows > BlockSize)
        {
            var rows = A.Rows - BlockSize;
            var cols = Math.Min(rows, BlockSize);
            Work = Matrix.Create(rows, cols, uninited: true);
        }

        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        while (A22.Rows > 0)
        {
            var block = Math.Min(BlockSize, A22.Rows);
            using var partAStep = partA.Step(block, block);

            LDLTLowerUnblock(A11);
            if (A22.Rows > 0)
            {
                TrSM(SideType.Right, UpLo.Lower, 
                    TransType.OnlyTrans, DiagType.Unit,
                    1.0, A11, A21);
                var Y21 = Work[..A21.Rows, ..A21.Cols];
                A21.CopyTo(Y21);
                for (var j = 0; j < block; j++)
                    A21[.., j].InvScaled(A11[j, j]);
                GeMMT(UpLo.Lower,
                    -1, Y21, A21.T, 1.0, A22);
            }
        }
    }

    internal static void LDLTUpperUnblock(MatrixView A)
    {
        int i = 0;
        ref var a11 = ref A[i, i];
        for (; i < A.Rows - 1; i++)
        {
            a11 = ref A.AtUncheck(i, i);
            if (a11 == 0)
                throw new LinalgException("Cholesky",
                    $"Warning: Cholesky factorization " +
                    $"encountered A non-positive pivot" +
                    $" at index {i} with value {a11}.");
            var a12 = A.SliceRowUncheck(i, i + 1);
            var A22 = A.SliceSubUncheck(i + 1, i + 1);
            InvScal(a11, a12);
            SyR(UpLo.Upper, -a11, a12, A22);
        }
    }

    internal static void LDLTUpperBlock(MatrixView A)
    {
        MatrixView Work = default;
        if (A.Rows > BlockSize)
        {
            var cols = A.Cols - BlockSize; 
            var rows = Math.Min(cols, BlockSize);
            Work = Matrix.Create(rows, cols, uninited: true);
        }

        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        while (A22.Rows > 0)
        {
            var block = Math.Min(BlockSize, A22.Rows);
            using var partAStep = partA.Step(block, block);

            LDLTUpperUnblock(A11);
            if (A22.Rows > 0)
            {
                TrSM(SideType.Left, UpLo.Upper,
                    TransType.OnlyTrans, DiagType.Unit,
                    1.0, A11, A12);
                var Y12 = Work[..A12.Rows, ..A12.Cols];
                A12.CopyTo(Y12);
                for (var i = 0; i < block; i++)
                    A12[i, ..].InvScaled(A11[i, i]);
                GeMMT(UpLo.Upper,
                    -1.0, A12.T, Y12, 1.0, A22);
            }
        }
    }

    internal static void LDLTSolveUncheck(UpLo uplo, MatrixView A, MatrixView X)
    {
        if (uplo == UpLo.Lower)
        {
            TrSM(SideType.Left, UpLo.Lower,
                TransType.NoTrans, DiagType.Unit,
                1, A, X);
            for(var i = 0; i < A.Rows; i++)
                X.GetRow(i).InvScaled(A[i, i]);
            TrSM(SideType.Left, UpLo.Upper,
                TransType.NoTrans, DiagType.Unit,
                1, A.T, X);
        }
        else if (uplo == UpLo.Upper)
        {
            TrSM(SideType.Left, UpLo.Lower,
                TransType.NoTrans, DiagType.Unit,
                1, A.T, X);
            for (var i = 0; i < A.Rows; i++)
                X.GetRow(i).InvScaled(A[i, i]);
            TrSM(SideType.Left, UpLo.Upper,
                TransType.NoTrans, DiagType.Unit,
                1, A, X);
        }
        else
        {
            throw new ArgumentException("Matrix must be either upper or lower triangular.");
        }
    }

}
