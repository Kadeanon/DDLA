using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using rscalar = double;
using DDLA.Misc.Flags;
using System.Runtime.CompilerServices;


namespace DDLA.BLAS;

public static partial class BlasProvider
{
    #region Checkers
    public static bool ShouldCheck { get; set; } = true;

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
        if (ShouldCheck)
        {
            if(uplo is not UpLo.Dense && m != n)
                throw new ArgumentException("Dimensions of matrixs a must be match!");
        }
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

    internal static (int rows, int cols) GetLengths(in matrix c)
    {
        var rows = c.Rows;
        var cols = c.Cols;
        return (rows, cols);
    }
    #endregion Getters

    public static void Asum(in vector x,
         out rscalar asum)
    {
        Source.Asum(
            x.Length,
            ref x.GetHeadRef(), x.Stride, out asum);
    }

    public static rscalar Nrm1(in matrix a, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(a, uplo);
        Source.Nrm1(0, 0,
            uplo,
            m, n,
            ref a.GetHeadRef(), a.RowStride, a.ColStride,
            out var norm);
        return norm;
    }

    public static rscalar NrmF(in matrix a, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(a, uplo);
        Source.NrmF(0, 0,
            uplo,
            m, n,
            ref a.GetHeadRef(), a.RowStride, a.ColStride,
            out var norm);
        return norm;
    }

    public static rscalar NrmInf(in matrix a, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(a, uplo);
        Source.NrmInf(0, 0,
            uplo,
            m, n,
            ref a.GetHeadRef(), a.RowStride, a.ColStride,
            out var norm);
        return norm;
    }

    public static rscalar Nrm1(in vector a)
    {
        Source.Nrm1(
            a.Length,
            ref a.GetHeadRef(), a.Stride,
            out var norm);
        return norm;
    }

    public static rscalar NrmF(in vector a)
    {
        Source.NrmF(
            a.Length,
            ref a.GetHeadRef(), a.Stride,
            out var norm);
        return norm;
    }

    public static rscalar NrmInf(in vector a)
    {
        Source.NrmInf(
            a.Length,
            ref a.GetHeadRef(), a.Stride,
            out var norm);
        return norm;
    }

    public static void MakeSy(in matrix a, UpLo uplo = UpLo.Lower)
    {
        int m = CheckSymmMatLength(a, uplo);
        Source.MkSym(uplo,
            m,
            ref a.GetHeadRef(), a.RowStride, a.ColStride);
    }

    public static void MakeTr(in matrix a, UpLo uplo = UpLo.Lower)
    {
        int m = CheckSymmMatLength(a, uplo);
        Source.MkTri(uplo,
            m,
            ref a.GetHeadRef(), a.RowStride, a.ColStride);
    }

    public static void Rand(in vector x)
    {
        Source.Rand(x.Length, ref x.GetHeadRef(), x.Stride);
    }

    public static void Rand(in matrix a, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = CheckUploMatLength(a, uplo);
        Source.Rand(0, 
            uplo,
            m, n,
            ref a.GetHeadRef(), a.RowStride, a.ColStride);
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
        in matrix a, in matrix b)
    {
        var (m, n) = CheckLength(a, aTrans, b);
        bool eq = false;
        Source.Eq(0, aDiag, aUpLo, aTrans,
            m, n,
            ref a.GetHeadRef(), a.RowStride, a.ColStride,
            ref b.GetHeadRef(), b.RowStride, b.ColStride,
            ref eq);
        return eq;
    }
}
