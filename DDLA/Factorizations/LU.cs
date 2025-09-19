using DDLA.BLAS;
using DDLA.Core;
using DDLA.Misc;
using DDLA.Misc.Flags;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using static DDLA.BLAS.BlasProvider;

namespace DDLA.Factorizations;

public class LU
{
    MatrixView matrix;
    readonly int[]? pivs;
    readonly int n;
    bool computed;
    Matrix? inverse;
    bool deconstructed;

    public Matrix Inverse
    {
        get
        {
            ComputeOnce();
            BuildInverseOnce();
            return inverse;
        }
    }

    public LU(MatrixView A, bool inplace = false, bool usePiv = true)
    {
        n = A.Rows;
        ArgumentOutOfRangeException.ThrowIfNotEqual(A.Cols, n, nameof(A));

        if (inplace)
            matrix = A;
        else
            matrix = A.Clone();

        pivs = usePiv ? new int[n] : null;
        computed = false;
    }

    private void ComputeOnce()
    {
        if (deconstructed)
            throw new InvalidOperationException(
                "Matrix has been deconstructed, cannot compute again.");
        if (computed)
            return;
        if (pivs == null)
        {
            LUDecompose(matrix);
        }
        else
        {
            PLUDecompose(matrix, pivs);
        }
        computed = true;
    }

    public Vector Solve(VectorView b, bool trans = false, bool inplace = false)
    {
        ComputeOnce();
        if (b.Length != n)
            throw new ArgumentException("RHS vector size mismatch", nameof(b));
        if (!inplace)
            b = b.Clone();
        if (pivs == null)
        {
            LUSolve(matrix, new(b), trans);
        }
        else
        {
            PLUSolve(matrix, pivs, new(b));
        }

        return new(b);
    }

    public Matrix Solve(MatrixView B, bool trans = false, bool inplace = false)
    {
        ComputeOnce();
        if (B.Rows != n)
            throw new ArgumentException("RHS vector size mismatch", nameof(B));
        if (!inplace)
            B = B.Clone();
        if (pivs == null)
        {
            LUSolve(matrix, B, trans);
        }
        else
        {
            PLUSolve(matrix, pivs, B, trans);
        }

        return new(B);
    }

    [MemberNotNull(nameof(inverse))]
    private void BuildInverseOnce()
    {
        if (inverse == null)
        {
            inverse = Matrix.Eyes(n);
            Solve(inverse, true);
        }
    }

    public void Deconstruct(out Matrix P, out Matrix L, out Matrix U)
    {
        ComputeOnce();

        P = Matrix.Eyes(n);
        if (pivs != null)
            ApplyPiv(P, pivs);

        L = matrix.Clone();
        MakeTr(L, UpLo.Lower);
        SetDiag(1.0, L);

        U = new(matrix);
        MakeTr(U, UpLo.Upper);

        deconstructed = true;
    }

    #region LU without pivoting
    internal static int LUBlockSize = 128;

    public static void LUDecompose(MatrixView A)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException("Matrix must be square for LU factorization.");
        if (length <= LUBlockSize)
            LUDecUnblock(A);
        else
            LUDecBlock(A);
    }

    public static void LUSolve(MatrixView A, MatrixView B, bool trans = false)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException("Matrix must be square for LU factorization.");
        if (length != B.Rows)
            throw new ArgumentException("RHS matrix size mismatch", nameof(B));
        LUSolveInternal(trans, A, B);
    }

    public static void LUSolve(MatrixView A, MatrixView B, MatrixView X, bool trans = false)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException("Matrix must be square for LU factorization.");
        if (length != B.Rows)
            throw new ArgumentException("RHS matrix size mismatch", nameof(B));
        if (length != X.Rows)
            throw new ArgumentException("Result matrix size mismatch", nameof(B));
        if (B.Cols != X.Cols)
            throw new ArgumentException("Result matrix size mismatch", nameof(B));
        B.CopyTo(X);
        LUSolveInternal(trans, A, X);
    }

    internal static void LUDecBlock(MatrixView A)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);
        while (A22.Rows > 0)
        {
            int block = Math.Min(LUBlockSize, A22.Rows);
            using var partAStep = partA.Step(block, block);

            LUDecUnblock(a11);
            TrSM(SideType.Left, UpLo.Lower, TransType.NoTrans, DiagType.Unit, 1, a11, a12);
            TrSM(SideType.Right, UpLo.Upper, 1, a11, a21);
            GeMM(-1, a21, a12, 1, A22);
        }
    }

    internal static void LUDecUnblock(MatrixView A)
    {
        int length = Math.Min(A.Rows, A.Cols);
        for (int i = 1; i < length; i++)
        {
            ref var alpha11 = ref A[0, 0];
            var a12t = A[0, 1..];
            var a21 = A[1.., 0];
            var A22 = A[1.., 1..];
            // a21 /= alpha11
            InvScal(alpha11, a21);
            // A22 -= a21 * a12t
            GeR(-1, a21, a12t, A22);
            A = A22;
        }
    }

    internal static void LUSolveInternal(bool trans, MatrixView A, MatrixView B)
    {
        if (trans)
        {
            // A^TX = B <=> (U^TL^T)X = B <=> U^T(L^TX) = B
            var LT = A;
            var UT = A;
            // Y <= Solve(U^TY = B)
            var Y = B;
            TrSM(SideType.Left, UpLo.Lower, TransType.OnlyTrans, DiagType.Unit, 1, UT, Y);
            // X <= Solve(L^TX = Y)
            var X = Y;
            TrSM(SideType.Left, UpLo.Upper, TransType.OnlyTrans, DiagType.NonUnit, 1, LT, X);
        }
        else
        {
            // AX = B <=> LUX = B <=> L(UX) = B
            var L = A;
            var U = A;
            // Y <= Solve(LY = B)
            var Y = B;
            TrSM(SideType.Left, UpLo.Lower, TransType.NoTrans, DiagType.Unit, 1, L, Y);
            // X <= Solve(UX = Y)
            var X = Y;
            TrSM(SideType.Left, UpLo.Upper, TransType.NoTrans, DiagType.NonUnit, 1, U, X);
        }
    }
    #endregion LU without pivoting

    #region LU with partial pivoting
    internal static int PLUBlockSize = 128;

    public static void ApplyPiv(MatrixView B, Span<int> pivs,
        SideType side = SideType.Left, bool forward = true)
    {
        if (side == SideType.Left && B.Rows != pivs.Length)
            throw new ArgumentException
                ("The length of pivs must match " +
                "the number of rows in the matrix.");
        else if (side == SideType.Right && B.Cols != pivs.Length)
            throw new ArgumentException
                ("The length of pivs must match " +
                "the number of columns in the matrix.");
        ApplyPivInternel(side, forward, B, pivs);
    }

    public static void PLUDecompose(MatrixView A, Span<int> pivs)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException(
                "Matrix must be square for LU factorization.");
        if (pivs.Length != length)
            throw new ArgumentException(
                "The length of pivs must match the number of rows in the matrix.");
        if (length >= PLUBlockSize)
            PLUDecBlock(A, pivs);
        else
            PLUDecUnblock(A, pivs);
    }

    public static void PLUSolve(MatrixView A, Span<int> pivs, MatrixView B,
        bool trans = false)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException("Matrix must be square for LU factorization.");
        if (length != B.Rows)
            throw new ArgumentException("RHS matrix size mismatch", nameof(B));
        PLUSolveInternel(trans, A, pivs, B);
    }

    public static void PLUSolve(MatrixView A, Span<int> pivs,
        MatrixView B, MatrixView X, bool trans = false)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException("Matrix must be square for LU factorization.");
        if (length != B.Rows)
            throw new ArgumentException("RHS matrix size mismatch", nameof(B));
        B.CopyTo(X);
        PLUSolveInternel(trans, A, pivs, X);
    }

    internal static void PLUDecBlock(MatrixView A, Span<int> p)
    {
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);
        var pPrev = p[..1];
        var pCurrent = p;
        var pNext = p;
        int block = Math.Min(PLUBlockSize, A22.Rows);

        while (A22.Rows > 0)
        {
            using var partAStep = partA.Step(block, block);
            {
                pCurrent = pNext[..block];
                pNext = pNext[block..];

                var rows = a11.Rows;
                PartUtils.Comb31to21(ref a01,
                                      in a11,
                                     ref a21, UpLo.Lower);
                PLUDecUnblock(a21, pCurrent);
                PartUtils.Part21to31(ref a01,
                                     out a11,
                                     ref a21, rows, UpLo.Lower);
                PartUtils.Comb31to21(ref A00,
                                      in a10,
                                     ref A20, UpLo.Lower);
                ApplyPivInternel(SideType.Left, true, A20, pCurrent);
                PartUtils.Part21to31(ref A00,
                                     out a10,
                                     ref A20, rows, UpLo.Lower);
                PartUtils.Comb31to21(ref A02,
                                      in a12,
                                     ref A22, UpLo.Lower);
                ApplyPivInternel(SideType.Left, true, A22, pCurrent);
                PartUtils.Part21to31(ref A02,
                                     out a12,
                                     ref A22, rows, UpLo.Lower);

                // A12 = trilu( A11 ) \ A12
                TrSM(SideType.Left, UpLo.Lower, TransType.NoTrans, DiagType.Unit, 1, a11, a12);
                // A22 -= A21 * A12
                GeMM(-1, a21, a12, 1, A22);
                pPrev = MemoryMarshal.CreateSpan(
                    ref MemoryMarshal.GetReference(p),
                    pPrev.Length + block);

                block = Math.Min(PLUBlockSize, A22.Rows);
            }
        }
    }

    internal static void PLUDecUnblock(MatrixView A, Span<int> pivs)
    {
        var full = A;
        int length = Math.Min(A.Cols, A.Rows);
        ref var p = ref pivs[0];

        for (int i = 0; i < length; i++)
        {
            p = GetAndApplyPivUnblock(A, full);
            ref var alpha11 = ref A[0, 0];
            var a12t = A[0, 1..];
            var a21 = A[1.., 0];
            var A22 = A[1.., 1..];
            // a21 /= alpha11 
            InvScal(alpha11, a21);
            // A22 -= a21 * a12t
            GeR(-1, a21, a12t, A22);
            A = A22;
            full = full.SliceSubUncheck(1, 0);
            p = ref Unsafe.Add(ref p, 1);
        }
    }

    internal static int GetAndApplyPivUnblock(MatrixView A, MatrixView full)
    {
        var col = A.GetColUncheck(0);
        var ipiv = AMax(col);
        if (ipiv != 0)
            full.SwapRow(0, ipiv);
        return ipiv;
    }

    internal static void ApplyPivInternel(SideType side, bool forward,
        MatrixView X, Span<int> pivs)
    {
        if (X.Rows == 0 || X.Cols == 0)
            return;
        if (side == SideType.Left)
        {
            if (forward) // X = PX
            {
                ref var p = ref pivs[0];
                for (int i = 0; i < pivs.Length; i++)
                {
                    if (p > 0)
                        X.SwapRow(i, i + p);
                    p = ref Unsafe.Add(ref p, 1);
                }
            }
            else // X = P^(-1)X
            {
                ref var p = ref pivs[pivs.Length - 1];
                for (var i = pivs.Length - 1;
                    i >= 0; i--)
                {
                    if (p > 0)
                        X.SwapRow(i, i + p);
                    p = ref Unsafe.Subtract(ref p, 1);
                }
            }
        }
        else
        {
            if (forward) // X = XP^T
            {
                ref var p = ref pivs[0];
                for (int i = 0; i < pivs.Length; i++)
                {
                    if (p > 0)
                        X.SwapCol(i, i + p);
                    p = ref Unsafe.Add(ref p, 1);
                }
            }
            else// X = X(P^T)^-1
            {
                ref var p = ref pivs[pivs.Length - 1];
                for (int i = pivs.Length - 1;
                    i >= 0; i--)
                {
                    if (p > 0)
                        X.SwapCol(i, i + p);
                    p = ref Unsafe.Subtract(ref p, 1);
                }
            }
        }
    }

    internal static void PLUSolveInternel(bool trans, MatrixView A,
        Span<int> pivs, MatrixView B)
    {
        if (trans)
        {
            // PA = LU <=> A^TP^T = U^TL^T <=> A^T = U^TL^TP
            // A^TX = B <=> (U^TL^TP)X = B <=> U^T(L^T(PX)) = B
            var LT = A;
            var UT = A;
            // V <= Solve(U^TV = B)
            var V = B;
            TrSM(SideType.Left, UpLo.Upper, TransType.OnlyTrans, DiagType.NonUnit, 1, UT, V);
            // Y <= Solve(L^TY = V)
            var Y = V;
            TrSM(SideType.Left, UpLo.Lower, TransType.OnlyTrans, DiagType.Unit, 1, LT, Y);
            // X <= Solve(PX = Y)
            var X = Y;
            ApplyPivInternel(SideType.Left, forward: false, X, pivs);
        }
        else
        {
            // AX = B <=> LUX = PB <=> L(UX) = (PB)
            var L = A;
            var U = A;
            // V = PB
            var V = B;
            ApplyPivInternel(SideType.Left, true, V, pivs);
            // Y <= Solve(L(UX) = V)
            var Y = V;
            TrSM(SideType.Left, UpLo.Lower, TransType.NoTrans, DiagType.Unit, 1, L, Y);
            // X <= Solve(UX = Y)
            var X = Y;
            TrSM(SideType.Left, UpLo.Upper, TransType.NoTrans, DiagType.NonUnit, 1, U, X);
        }
    }
    #endregion LU with partial pivoting
}