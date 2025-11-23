using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using rscalar = double;
using DDLA.Misc.Flags;
using System.Runtime.CompilerServices;
using EnumsNET;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    #region Checkers
    public static bool ShouldCheck { get; set; } = true;

    internal static void CheckLengths(in matrix mat, int m0, int n0)
    {
        int m = mat.Rows;
        int n = mat.Cols;
        if (ShouldCheck && m != m0 || n != n0)
            throw new ArgumentException($"Dimensions of matrixs must be match! Expected: ({m0}, {n0}), Actual: ({m}, {n})");
    }

    internal static int CheckSymmMatLength(in matrix mat, UpLo uplo)
    {
        int m = mat.Rows;
        if (ShouldCheck && mat.Cols != m)
            throw new ArgumentException("Dimensions of matrixs a must be match!");
        if (ShouldCheck && uplo != UpLo.Upper && uplo != UpLo.Lower)
            throw new ArgumentException($"Matrix c must be upper or lower triangular!");
        return m;
    }

    internal static (int m, int n) CheckUploMatLength(in matrix mat, UpLo uplo)
    {
        int m = mat.Rows;
        int n = mat.Cols;
        //if (ShouldCheck)
        //{
        //    if(uplo is not UpLo.Dense && m != n)
        //        throw new ArgumentException("Dimensions of matrixs a must be match!");
        //}
        return (m, n);
    }

    internal static TransType CheckLengthsAfterTrans(in matrix mat, TransType trans, int m0, int n0)
    {
        int m = mat.Rows;
        int n = mat.Cols;
        if ((trans & TransType.OnlyTrans) == TransType.OnlyTrans)
            (m, n) = (n, m);
        if (ShouldCheck && m != m0 || n != n0)
            throw new ArgumentException($"Dimensions of matrixs must be match! Expected: ({m0}, {n0}), Actual: ({m}, {n})");
        return trans;
    }

    internal static void CheckLength(in vector vec, int expectedLength)
    {
        if (ShouldCheck && vec.Length != expectedLength)
            throw new ArgumentException($"Length of vector must be equal to {expectedLength}.");
    }

    internal static int CheckLength(in vector x, in vector y)
    {
        if (ShouldCheck && x.Length != y.Length)
            throw new ArgumentException("Error: x and y must have the same length.");
        return x.Length;
    }

    internal static int CheckLength(in vector x, in vector y, in vector z)
    {
        if (ShouldCheck)
        {
            if (x.Length != y.Length)
                throw new ArgumentException("Error: x and y must have the same length.");
            if (x.Length != z.Length)
                throw new ArgumentException("Error: x and z must have the same length.");
        }
        return x.Length;
    }

    internal static (int m, int n) CheckLength(in matrix a, TransType aTrans, in matrix b)
    {
        var m = a.Rows;
        var n = a.Cols;
        if (aTrans.HasFlag(TransType.OnlyTrans))
            (m, n) = (n, m);
        if (ShouldCheck && m != b.Rows || n != b.Cols)
            throw new ArgumentException("Matrix dimensions do not match.");
        return (m, n);
    }
    #endregion Checkers

    #region Getters
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (int m, int n) GetLengthsAfterTrans(in matrix mat, TransType trans)
    {
        int m = mat.Rows;
        int n = mat.Cols;
        if ((trans & TransType.OnlyTrans) == TransType.OnlyTrans)
            return (n, m);
        else
            return (m, n);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (int m, int n) GetStridesAfterTrans(in matrix mat, TransType trans)
    {
        int m = mat.RowStride;
        int n = mat.ColStride;
        if ((trans & TransType.OnlyTrans) == TransType.OnlyTrans)
            return (n, m);
        else
            return (m, n);
    }

    internal static (int rows, int cols) GetLengths(in matrix c)
    {
        var rows = c.Rows;
        var cols = c.Cols;
        return (rows, cols);
    }
    #endregion Getters

    public static UpLo Transpose(UpLo upLo)
        => upLo switch
        {
            UpLo.Upper => UpLo.Lower,
            UpLo.Lower => UpLo.Upper,
            _ => upLo
        };

    public static TransType Transpose(TransType trans)
        => trans.HasFlag(TransType.OnlyTrans) ?
        trans.RemoveFlags(TransType.OnlyTrans) : 
        trans.CombineFlags(TransType.OnlyTrans);

    public static void Asum(in vector x, out rscalar asum)
        => asum = UFunc.Sum<AbsOperator<double>>(x);

    public static rscalar Nrm1(in matrix A, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(A, uplo);
        Source.Nrm1(0, 0,
            uplo,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            out var norm);
        return norm;
    }

    public static rscalar NrmF(in matrix A, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(A, uplo);
        Source.NrmF(0, 0,
            uplo,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            out var norm);
        return norm;
    }

    public static rscalar NrmInf(in matrix A, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(A, uplo);
        Source.NrmInf(0, 0,
            uplo,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            out var norm);
        return norm;
    }

    public static rscalar Nrm1(in vector x)
    {
        Source.Nrm1(
            x.Length,
            ref x.GetHeadRef(), x.Stride,
            out var norm);
        return norm;
    }

    public static rscalar NrmF(in vector x)
    {
        Source.NrmF(
            x.Length,
            ref x.GetHeadRef(), x.Stride,
            out var norm);
        return norm;
    }

    public static rscalar NrmInf(in vector x)
    {
        Source.NrmInf(
            x.Length,
            ref x.GetHeadRef(), x.Stride,
            out var norm);
        return norm;
    }

    public static void MakeSy(in matrix A, UpLo uplo = UpLo.Lower)
    {
        int m = CheckSymmMatLength(A, uplo);
        Copy(DiagType.Unit, Transpose(uplo), TransType.OnlyTrans, A, A);
    }

    public static void MakeTr(in matrix A, UpLo uplo = UpLo.Lower)
    {
        int m = CheckSymmMatLength(A, uplo);
        Set(DiagType.Unit, Transpose(uplo), 0.0, A);
    }

    public static void Rand(in vector x)
    {
        Source.Rand(x.Length, ref x.GetHeadRef(), x.Stride);
    }

    public static void Rand(in matrix A, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(A, uplo);
        Source.Rand(0,
            uplo,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void Sumsq(in vector x, ref rscalar sumsq, ref rscalar scale)
    {
        Source.SumSq(x.Length,
            ref x.GetHeadRef(), x.Stride,
            ref sumsq, ref scale);
    }

    public static bool Equals(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        bool eq = false;
        Source.Eq(ConjType.NoConj,
            length,
            ref x.GetHeadRef(), x.Stride,
            ref y.GetHeadRef(), y.Stride,
            ref eq);
        return eq;
    }

    public static bool Equals(DiagType aDiag, UpLo aUpLo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        bool eq = false;
        Source.Eq(0, aDiag, aUpLo, aTrans,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            ref eq);
        return eq;
    }
}
