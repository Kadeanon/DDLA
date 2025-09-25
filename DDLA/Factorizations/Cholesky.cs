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

    public Cholesky(MatrixView mat, UpLo uplo = UpLo.Lower, bool inplace = false)
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

    public Vector Solve(VectorView b, bool inplace = false)
    {
        ComputeOnce();
        var bMat = new MatrixView(b);
        if (!inplace)
            bMat = bMat.Clone();
        CholeskySolve(uplo, matrix, bMat);
        return new(bMat.GetColumn(0));
    }

    public Matrix Solve(MatrixView b, bool inplace = false)
    {
        ComputeOnce();
        if (!inplace)
            b = b.Clone();
        CholeskySolve(uplo, matrix, b);
        return new(b);
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

        for (var i = 0; i < A.Rows; i++)
        {
            ref var alpha11 = ref A[i, i];
            var a21 = A[(i + 1).., i];
            var A22 = A[(i + 1).., (i + 1)..];
            Sqrt(ref alpha11);
            a21.InvScaled(alpha11);
            A22.Rank1(UpLo.Lower, -1, a21);
        }
    }

    internal static void CholeskyLowerBlock(MatrixView A)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);

        while (A22.Rows > 0)
        {
            var block = Math.Min(BlockSize, A22.Rows);
            using var partAStep = partA.Step(block, block);

            CholeskyLowerUnblock(a11);
            TrSM(SideType.Right, UpLo.Lower, TransType.OnlyTrans, DiagType.NonUnit,
                1, a11, a21);
            SyRk(UpLo.Lower, TransType.NoTrans, -1, a21, 1, A22);
        }
    }

    internal static void CholeskyUpperUnblock(MatrixView A)
    {
        if (A.IsEmpty) return;

        for (var i = 0; i < A.Rows; i++)
        {
            ref var alpha11 = ref A[i, i];
            var a01 = A[..i, i];
            var A02 = A[..i, (i + 1)..];
            var a12 = A[i, (i + 1)..];

            alpha11 -= a01 * a01;
            alpha11 = Math.Sqrt(alpha11);
            // a12' -= A02' * a01v
            //GeMV(-1, A02.T, a01, 1, a12);
            a01.LeftMul(-1.0, A02, 1.0, a12);
            // a12t /= sqrt(alpha11)
            a12.InvScaled(alpha11);

            //alpha11 -= a01.SumSq();
            //Sqrt(ref alpha11);
            //a01.LeftMul(-1, A02, 1, a12);
            //a12.InvScaled(alpha11);
        }
    }

    internal static void CholeskyUpperBlock(MatrixView A)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);

        while (A22.Rows > 0)
        {
            var block = Math.Min(BlockSize, A22.Rows);
            using var partAStep = partA.Step(block, block);

            // A11 = chol( A11 )
            CholeskyUpperUnblock(a11);
            // A12 = inv( triu( A11 )' ) * A12
            TrSM(SideType.Left, UpLo.Upper, TransType.OnlyTrans, DiagType.NonUnit, 1, a11, a12);
            // A22 -= A12' * A12
            SyRk(UpLo.Upper, -1, a12.T, 1, A22);
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

    private static void Sqrt(ref double alpha)
    {
        if (alpha <= 0)
            throw new LinalgException("Cholesky",
                "Warning: Cholesky factorization " +
                "encountered A non-positive pivot" +
                $"with value {alpha}.");
        alpha = Math.Sqrt(alpha);
    }

}
