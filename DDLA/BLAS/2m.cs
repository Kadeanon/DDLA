using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;
using System.Runtime.CompilerServices;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    public static void GeMV
        (TransType aTrans, scalar alpha, 
        in matrix A, in vector x, scalar beta, in vector y)
    {
        var (m, n) = GetLengthsAfterTrans(A, aTrans);
        CheckLength(x, n);
        CheckLength(y, m);
        if (m == 0 || n == 0) 
            return;

        Source.GeMV(aTrans, 
            ConjType.NoConj, 
            A.Rows, A.Cols, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride, 
            ref x.GetHeadRef(), x.Stride, 
            beta, 
            ref y.GetHeadRef(), y.Stride);
    }

    public static void GeMV
        (scalar alpha, in matrix A, in vector x, scalar beta, in vector y)
        => GeMV(TransType.NoTrans, alpha, A, x, beta, y);

    public static void GeR
        (scalar alpha, in vector x, in vector y, in matrix A)
    {
        var (m, n) = GetLengths(A);
        CheckLength(x, m);
        CheckLength(y, n);

        Source.GeR(ConjType.NoConj, 
            ConjType.NoConj, 
            m, n, 
            in alpha, 
            ref x.GetHeadRef(), x.Stride, 
            ref y.GetHeadRef(), y.Stride, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void SyMV
        (UpLo aUplo, scalar alpha, in matrix A, in vector x, scalar beta, in vector y)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(x, length);
        CheckLength(y, length);

        Source.SyMV(aUplo,
            ConjType.NoConj,
            ConjType.NoConj, 
            length, 
            in alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride, 
            ref x.GetHeadRef(), x.Stride, 
            beta, 
            ref y.GetHeadRef(), y.Stride);
    }

    public static void SyR
        (UpLo aUplo, scalar alpha, in vector x, in matrix A)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(x, length);

        Source.SyR(aUplo, 
            ConjType.NoConj, 
            length, 
            in alpha,
            ref x.GetHeadRef(), x.Stride,
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void SyR2
        (UpLo aUplo, scalar alpha, in vector x, in vector y, in matrix A)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(x, length);
        CheckLength(y, length);

        Source.SyR2(aUplo, 
            ConjType.NoConj, 
            ConjType.NoConj, 
            length, 
            in alpha, 
            ref x.GetHeadRef(), x.Stride,
            ref y.GetHeadRef(), y.Stride,
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void TrMV
        (UpLo aUplo, TransType aTrans, DiagType aDiag, scalar alpha, in matrix A, in vector x)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(x, length);

        Source.TrMV(aUplo,
            aTrans,
            aDiag,
            length,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref x.GetHeadRef(), x.Stride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TrMV
        (UpLo aUplo, scalar alpha, in matrix A, in vector y)
        => TrMV(aUplo, TransType.NoTrans, DiagType.NonUnit,
            alpha, A, y);

    public static void TrSV
        (UpLo aUplo, TransType aTrans, DiagType aDiag, scalar alpha, in matrix A, in vector y)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(y, length);

        Source.TrSV(aUplo,
            aTrans,
            aDiag,
            length,
            in alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref y.GetHeadRef(), y.Stride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TrSV
        (UpLo aUplo, scalar alpha, in matrix A, in vector y) 
        => TrSV(aUplo, TransType.NoTrans, DiagType.NonUnit,
            alpha, A, y);
}
