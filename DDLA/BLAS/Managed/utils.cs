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
            UpLo.Dense => UpLo.Dense,
        };

    public static TransType Transpose(TransType trans)
        => trans.HasFlag(TransType.OnlyTrans) ?
        trans.RemoveFlags(TransType.OnlyTrans) : 
        trans.CombineFlags(TransType.OnlyTrans);

    public static SideType Transpose(SideType side)
        => side switch
        {
            SideType.Left => SideType.Right,
            SideType.Right => SideType.Left,
        };

    public static void Asum(in vector x, out rscalar asum)
    {
        asum = 0.0;
        foreach(var val in x)
        {
            var abs = Math.Abs(val);
            if(double.IsNaN(abs) || double.IsInfinity(abs))
            {
                asum = abs;
                return;
            }
            asum += abs;
        }
    }

    public static rscalar Nrm1(in matrix A, UpLo uplo = UpLo.Dense)
    {
        var (m, n) = GetLengths(A);

        rscalar max = 0.0;

        if (uplo == UpLo.Dense)
        {
            for (var j = 0; j < n; j++)
                max = Math.Max(max, Nrm1(A.GetColUncheck(j))); 
            return max;
        }

        for (var j = 0; j < n; j++)
        {
            rscalar colsum = 0.0;

            if (uplo == UpLo.Upper)
            {
                // upper: i <= j
                var iMax = Math.Min(m - 1, j);
                for (var i = 0; i <= iMax; i++)
                    colsum += Math.Abs(A[i, j]);
            }
            else // Lower
            {
                // lower: i >= j
                var iMin = Math.Max(0, j);
                for (var i = iMin; i < m; i++)
                    colsum += Math.Abs(A[i, j]);
            }

            if (colsum > max) max = colsum;
        }

        return max;
    }

    public static rscalar NrmF(in matrix A, UpLo uplo = UpLo.Dense)
    {
        rscalar scale = 0.0;
        rscalar sumsq = 1.0;
        var AEffective = A;
        if (A.RowStride < A.ColStride)
        {
            AEffective = AEffective.T;
            uplo = Transpose(uplo);
        }
        var (m, _) = GetLengths(AEffective);
        if (uplo == UpLo.Dense)
        {
            for (var i = 0; i < m; i++)
            SumSq(AEffective.GetRowUncheck(i), ref sumsq, ref scale);
        }
        else if (uplo == UpLo.Upper)
        {
            var minDim = AEffective.MinDim;
            for (var i = 0; i < minDim; i++)
                SumSq(AEffective.SliceRowUncheck(i, i), ref sumsq, ref scale);
        }
        else // if (uplo == UpLo.Upper)
        {
            var minDim = AEffective.MinDim;
            var i = 0;
            for (; i < minDim; i++)
                SumSq(AEffective.SliceRowUncheck(i, 0, i + 1), ref sumsq, ref scale);
            for (; i < m; i++)
                SumSq(AEffective.GetRowUncheck(i), ref sumsq, ref scale);
        }
        return (scale == 0.0) ? 0.0 : scale * Math.Sqrt(sumsq);
    }

    public static rscalar NrmInf(in matrix A, UpLo uplo = UpLo.Dense)
    {
        return Nrm1(A.T, Transpose(uplo));
    }

    public static rscalar Nrm1(in vector x)
    {
        Asum(x, out var result);
        return result;
    }

    public static rscalar NrmF(in vector x)
    {
        rscalar scale = 0.0;
        rscalar sumsq = 1.0;
        SumSq(x, ref sumsq, ref scale);
        return scale * Math.Sqrt(sumsq);
    }

    public static rscalar NrmInf(in vector x)
    {
        var max = 0.0;
        foreach (var val in x)
        {
            var abs = Math.Abs(val);
            if (double.IsNaN(abs) || double.IsInfinity(abs))
            {
                max = abs;
                break;
            }
            max = Math.Max(abs, max);
        }
        return max;
    }

    public static void MakeSy(in matrix A, UpLo uplo = UpLo.Lower)
    {
        CheckSymmMatLength(A, uplo);
        Copy(DiagType.Unit, Transpose(uplo), TransType.OnlyTrans, A.T, A.T);
    }

    public static void MakeTr(in matrix A, UpLo uplo = UpLo.Lower)
    {
        CheckSymmMatLength(A, uplo);
        Set(DiagType.Unit, Transpose(uplo), 0.0, A);
    }

    public static void Rand(in vector x, Random? random = null)
    {
        if (x.Length == 0) return;
        random ??= new Random();
        foreach (ref var val in x)
            val = random.NextDouble();
    }

    public static void Rand(in matrix A, UpLo uplo = UpLo.Dense, 
        Random? random = null)
    {
        var AEffective = A;
        var (m, _) = GetLengths(AEffective);
        if (A.RowStride < A.ColStride)
        {
            AEffective = AEffective.T;
            uplo = Transpose(uplo);
        }
        if (uplo == UpLo.Dense)
        {
            for (var i = 0; i < m; i++)
                Rand(AEffective.GetRowUncheck(i), random);
        }
        else if(uplo == UpLo.Upper)
        {
            var minDim = AEffective.MinDim;
            for (var i = 0; i < minDim; i++)
                Rand(AEffective.SliceRowUncheck(i, i), random);
        }
        else // if (uplo == UpLo.Upper)
        {
            var minDim = AEffective.MinDim;
            var i = 0;
            for (; i < minDim; i++)
                Rand(AEffective.SliceRowUncheck(i, 0, i), random);
            for (; i < m; i++)
                Rand(AEffective.GetRowUncheck(i), random);
        }
    }

    public static void SumSq(in vector x, ref rscalar sumsq, ref rscalar scale)
    {
        if (x.Length == 0) return;
        foreach (ref var val in x)
        {
            rscalar abs = Math.Abs(val);

            if (double.IsNaN(abs) || double.IsInfinity(abs))
            {
                sumsq = abs;
                scale = 1.0;
                break;
            }

            if (abs > 0.0)
            {
                if (scale < abs)
                {
                    var r = scale / abs;   
                    sumsq = 1.0 + sumsq * r * r;
                    scale = abs;
                }
                else
                {
                    var r = abs / scale;  
                    sumsq += r * r;
                }
            }
        }
    }

    public static bool Equals(in vector x, in vector y, double eps = 2e-16)
    {
        int length = CheckLength(x, y);
        bool eq = false;
        // TODO: use SIMD
        for(var i = 0; i < length; i++)
        {
            eq &= Math.Abs(x[i] - y[i]) < eps;
        }
        return eq;
    }

    public static bool Equals(DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B, double eps = 2e-16)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return true;
        var AEffective = A;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        var BEffective = B;
        if (B.RowStride < B.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            for (int i = 0; i < m; i++)
            {
                var rowA = AEffective.GetRowUncheck(i);
                var rowB = BEffective.GetRowUncheck(i);
                if (!Equals(rowA, rowB, eps)) return false;
            }
        }
        else if (aUplo is UpLo.Upper)
        {
            for (int i = 0; i < m; i++)
            {
                var start = i;
                if (aDiag is DiagType.Unit)
                    start++;
                start = Math.Max(start, 0);
                if (start >= n)
                    break;
                var rowA = AEffective.GetRowUncheck(i);
                var rowB = BEffective.GetRowUncheck(i);
                if (!Equals(rowA, rowB, eps)) return false;
            }
        }
        else // if (aUplo is UpLo.Lower)
        {
            for (int i = 0; i < m; i++)
            {
                var diagBound = aDiag is DiagType.Unit ? i : i + 1;
                var end = Math.Min(diagBound, n);
                if (end <= 0)
                    continue;
                var rowA = AEffective.SliceRowUncheck(i, 0, end);
                var rowB = BEffective.SliceRowUncheck(i, 0, end);
                if (!Equals(rowA, rowB, eps)) return false;
            }
        }
        return true;
    }
}
