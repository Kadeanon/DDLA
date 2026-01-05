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
        Blis.Add(0, aDiag, aTrans, 
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
        Blis.Axpy(0, aDiag, aTrans, 
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
        Blis.Copy(0, aDiag, aTrans, 
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
        Blis.Invert(0, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void InvscalDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        Blis.Scal(ConjType.NoConj, 0, 
            m, n, 
            1 / alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void ScalDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        Blis.Scal(ConjType.NoConj, 0, 
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void Scal2Diag
        (DiagType aDiag, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Blis.Scal2(0, aDiag, aTrans, 
            m, n, 
            in alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    public static void SetDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        Blis.Set(ConjType.NoConj, 0, 
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void ShiftDiag(scalar alpha, in matrix A)
    {
        var (m, n) = GetLengths(A);
        Blis.Shift(0, 
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static void SubDiag
        (DiagType aDiag, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Blis.Sub(0, aDiag, aTrans, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    public static void XpbyDiag
        (DiagType aDiag, TransType aTrans,
        in matrix A, scalar beta, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        Blis.Xpby(0, aDiag, aTrans, 
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            beta, 
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }
}
