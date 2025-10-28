using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    /// <summary>
    /// B += Trans?(A)
    /// </summary>
    public static void Add(DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Add(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride, 
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B += Trans?(A)
    /// </summary>
    public static void Add(in matrix A, in matrix B)
        => Add(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

    /// <summary>
    /// B += alpha * Trans?(A)
    /// </summary>
    public static void Axpy
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Axpy(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride, 
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B += alpha * Trans?(A)
    /// </summary>
    public static void Axpy(scalar alpha, 
        in matrix A, in matrix B)
        => Axpy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// B = Trans?(A)
    /// </summary>
    public static void Copy
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Copy(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride, 
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B = Trans?(A)
    /// </summary>
    public static void Copy(in matrix A, in matrix B)
        => Copy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

    /// <summary>
    /// A = A / alpha
    /// </summary>
    public static void InvScal(UpLo aUplo, scalar alpha, in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        if (m == 0 || n == 0) return;
        Source.Scal(ConjType.NoConj, A.DiagOffset, aUplo,
            m, n,
            1 / alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    /// <summary>
    /// A = A / alpha
    /// </summary>
    public static void InvScal(scalar alpha,
        in matrix A)
        => InvScal(UpLo.Dense, alpha, A);

    /// <summary>
    /// A = alpha * A
    /// </summary>
    public static void Scal(UpLo aUplo, scalar alpha, in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        if (m == 0 || n == 0) return;
        Source.Scal(ConjType.NoConj, A.DiagOffset, aUplo,
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    /// <summary>
    /// A = alpha * A
    /// </summary>
    public static void Scal(scalar alpha,
        in matrix A)
        => Scal(UpLo.Dense, alpha, A);

    /// <summary>
    /// B = alpha * Trans?(A)
    /// </summary>
    public static void Scal2
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Scal2(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B = alpha * Trans?(A)
    /// </summary>
    public static void Scal2(double alpha, in matrix A, in matrix B)
        => Scal2(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// B = alpha
    /// </summary>
    public static void Set
        (DiagType aDiag, UpLo aUplo, scalar alpha, in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        if (m == 0 || n == 0) return;
        Source.Set(ConjType.NoConj, A.DiagOffset, aDiag, aUplo,
            m, n, 
            alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    /// <summary>
    /// B = alpha
    /// </summary>
    public static void Set(scalar alpha,
        in matrix A)
        => Set(DiagType.NonUnit, UpLo.Dense, 
            alpha, A);

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Sub(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub(in matrix A, in matrix B)
        => Sub(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

}
