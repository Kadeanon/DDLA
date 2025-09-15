using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    /// <summary>
    /// B.Diag = Trans?(A.Diag)
    /// </summary>
    public static void AddDiag
        (DiagType aDiag, TransType aTrans, 
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Source.Add(A.DiagOffset, aDiag, aTrans, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B.Diag += alpha * Trans?(A.Diag)
    /// </summary>
    public static void AxpyDiag
        (DiagType aDiag, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Source.Axpy(A.DiagOffset, aDiag, aTrans, 
            m, n, 
            in alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B.Diag = Trans?(A.Diag)
    /// </summary>
    public static void CopyDiag
        (DiagType aDiag, TransType aTrans, 
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Source.Copy(A.DiagOffset, aDiag, aTrans, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B.Diag = one / B.Diag
    /// </summary>
    /// <param name="A"></param>
    public static void InvertDiag(in matrix A)
    {
        var (m, n) = GetLengths(A);
        Source.Invert(A.DiagOffset, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void InvscalDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        Source.Scal(ConjType.NoConj, A.DiagOffset, 
            m, n, 
            1 / alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void ScalDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        Source.Scal(ConjType.NoConj, A.DiagOffset, 
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void Scal2Diag
        (DiagType aDiag, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Source.Scal2(A.DiagOffset, aDiag, aTrans, 
            m, n, 
            in alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    public static void SetDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        Source.Set(ConjType.NoConj, A.DiagOffset, 
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void ShiftDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        Source.Shift(A.DiagOffset, 
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void SubDiag
        (DiagType aDiag, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Source.Sub(A.DiagOffset, aDiag, aTrans, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    public static void XpbyDiag
        (DiagType aDiag, TransType aTrans,
        in matrix A, scalar beta, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Source.Xpby(A.DiagOffset, aDiag, aTrans, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            beta, 
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }
}
