using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    /// <summary>
    /// 
    /// </summary>
    public static void Axpy2(scalar alphax, scalar alphay, in vector x, in vector y, in vector z)
    {
        int length = CheckLength(x, y, z);
        if (length == 0) return;
        Source.Axpy2V(ConjType.NoConj, ConjType.NoConj,
            length, in alphax, in alphay,
            ref x.GetHeadRef(), x.Stride,
            ref y.GetHeadRef(), y.Stride,
            ref z.GetHeadRef(), z.Stride);
    }

    public static void DotAxpy(scalar alpha, in vector x, in vector y, ref scalar rho, in vector z)
    {
        int length = CheckLength(x, y, z);
        if (length == 0) return;
        Source.DotAxpyV(ConjType.NoConj, ConjType.NoConj, ConjType.NoConj,
            length, in alpha,
            ref x.GetHeadRef(), x.Stride,
            ref y.GetHeadRef(), y.Stride,
            ref rho,
            ref z.GetHeadRef(), z.Stride);
    }

    public static void AxpyF(scalar alpha, in matrix A, in vector x, in vector y)
    {
        var (m, n) = GetLengths(A);
        CheckLength(x, n);
        CheckLength(y, m);
        Source.AxpyF(ConjType.NoConj, ConjType.NoConj,
            m, n,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref x.GetHeadRef(), x.Stride,
            ref y.GetHeadRef(), y.Stride);
    }

    public static void DotxF(scalar alpha, in matrix A, in vector x, scalar beta, in vector y)
    {
        var (m, n) = GetLengths(A);
        CheckLength(x, m);
        CheckLength(y, n);
        Source.DotxF(ConjType.NoConj, ConjType.NoConj,
            m, n,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref x.GetHeadRef(), x.Stride,
            in beta,
            ref y.GetHeadRef(), y.Stride);
    }

    public static void DotxAxpyF(scalar alpha, in matrix A, in vector w, in vector x,
        scalar beta, in vector y, in vector z)
    {
        var (m, n) = GetLengths(A);
        CheckLength(w, m);
        CheckLength(x, n);
        CheckLength(y, n);
        CheckLength(z, m);
        Source.DotxAxpyF(ConjType.NoConj, 
            ConjType.NoConj,
            ConjType.NoConj,
            ConjType.NoConj,
            m, n,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref w.GetHeadRef(), w.Stride,
            ref x.GetHeadRef(), x.Stride,
            in beta,
            ref y.GetHeadRef(), y.Stride,
            ref z.GetHeadRef(), z.Stride);
    }

}
