using DDLA.Core;
using DDLA.Misc;
using DDLA.Misc.Flags;
using System.Runtime.CompilerServices;

using static DDLA.BLAS.BlasProvider;

namespace DDLA.Factorizations;

public class Cholesky
{
    readonly UpLo uplo;
    readonly MatrixView matrix;
    bool computed;

    public Cholesky(UpLo uplo, MatrixView mat, bool inplace = false)
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
        CholDecompose(uplo, matrix);
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
    public static void CholDecompose(UpLo uplo, MatrixView A)
    {
        CheckSymmMatLength(A, uplo);
        if (uplo == UpLo.Lower)
        {
            if (A.MinDim > BlockSize)
                CholeskyLowerBlock(A);
            else
                CholeskyLowerUnblock(A);
        }
        else
        {
            if (A.MinDim > BlockSize)
                CholeskyUpperBlock(A);
            else
                CholeskyUpperUnblock(A);
        }
    }

    public VectorView Solve(VectorView b, bool inplace = false)
    {
        ComputeOnce();
        var bMat = new MatrixView(b);
        if (!inplace)
            bMat = bMat.Clone();
        CholeskySolve(uplo, matrix, bMat);
        return bMat.GetColumn(0);
    }

    public MatrixView Solve(MatrixView b, bool inplace = false)
    {
        ComputeOnce();
        if (!inplace)
            b = b.Clone();
        CholeskySolve(uplo, matrix, b);
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
    public static void CholeskySolve(UpLo uplo, MatrixView A,
        MatrixView B)
    {
        var m = CheckSymmMatLength(A, uplo);
        if (B.Rows != m)
            throw new ArgumentException(
                "Matrix dimensions do not match for Cholesky solve.");

        CholeskySolveUncheck(uplo, A, B);
    }

    public static void CholeskySolve(UpLo uplo, MatrixView A,
        MatrixView B, MatrixView X)
    {
        var m = CheckSymmMatLength(A, uplo);
        if (X.Rows != m)
            throw new ArgumentException(
                "Matrix dimensions do not match for Cholesky solve.");
        B.CopyTo(X);

        CholeskySolveUncheck(uplo, A, X);
    }

    internal static int BlockSize = 128;

    internal static void CholeskyLowerUnblock(MatrixView A)
    {
        if (A.IsEmpty) return;

        int i = 0;
        ref var a11 = ref A.GetHeadRef();
        var diagStride = A.RowStride + A.ColStride;
        for (; i < A.Rows - 1; i++)
        {
            var a21 = A.SliceColUncheck(i, i + 1);
            var a22 = A.SliceSubUncheck(i + 1, i + 1);
            if (a11 <= 0)
                throw new LinalgException("Cholesky",
                    $"Warning: Cholesky factorization " +
                    $"encountered A non-positive pivot" +
                    $" at index {i} with value {a11}.");
            a11 = Math.Sqrt(a11);
            InvScal(a11, a21);
            SyR(UpLo.Lower, -1, a21, a22);
            a11 = ref Unsafe.Add(ref a11, diagStride);
        }
        a11 = Math.Sqrt(a11);
    }

    internal static void CholeskyLowerBlock(MatrixView A)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var a00, out var a01, out var a02,
            out var a10, out var a11, out var a12,
            out var a20, out var a21, out var a22);
        var block = Math.Min(BlockSize, a22.Rows);

        while (a22.Rows > 0)
        {
            using var partAStep = partA.Step(block, block);

            CholeskyLowerUnblock(a11);
            TrSM(SideType.Right, UpLo.Lower, TransType.OnlyTrans, DiagType.NonUnit,
                1, a11, a21);
            SyRk(UpLo.Lower, TransType.NoTrans, -1, a21, 1, a22);
            block = Math.Min(BlockSize, a22.Rows);
        }
    }

    internal static void CholeskyUpperUnblock(MatrixView A)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var a00, out var a01, out var a02,
            out var a10, out var a11, out var a12,
            out var a20, out var a21, out var a22);

        while (a22.Rows > 0)
        {
            using var partAStep = partA.Step();

            ref var alpha11 = ref a11.GetHeadRef();
            var a01v = a01.GetColumn(0);
            var a12t = a12.GetRow(0);
            var a02t = a02.T;

            // alpha11 -= a01' * a01
            alpha11 -= a01v * a01v;
            alpha11 = Math.Sqrt(alpha11);
            // a12' -= A02' * a01v
            GeMV(-1, a02t, a01v, 1, a12t);
            // a12t /= sqrt(alpha11)
            InvScal(alpha11, a12t);
        }
    }

    internal static void CholeskyUpperBlock(MatrixView A)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);
        int block = Math.Min(BlockSize, A22.Rows);

        while (block > 0)
        {
            using var partAStep = partA.Step(block, block);

            // A11 = chol( A11 )
            CholeskyUpperUnblock(a11);
            // A12 = inv( triu( A11 )' ) * A12
            TrSM(SideType.Left, UpLo.Upper, TransType.OnlyTrans, DiagType.NonUnit, 1, a11, a12);
            // A22 -= A12' * A12
            SyRk(UpLo.Upper, -1, a12.T, 1, A22);

            block = Math.Min(BlockSize, A22.Rows);
        }
    }

    internal static void CholeskySolveUncheck(UpLo uplo, MatrixView A, MatrixView X)
    {
        if (uplo == UpLo.Lower)
        {
            TrSM(SideType.Left, UpLo.Lower, 1, A, X);
            TrSM(SideType.Left, UpLo.Upper, 1, A.T, X);
        }
        else if (uplo == UpLo.Upper)
        {
            TrSM(SideType.Left, UpLo.Lower, 1, A.T, X);
            TrSM(SideType.Left, UpLo.Upper, 1, A, X);
        }
        else
        {
            throw new ArgumentException("Matrix must be either upper or lower triangular.");
        }
    }

    internal static void CholeskyInverse(MatrixView A, MatrixView I)
    {
        A.CopyTo(I);
    }
}
