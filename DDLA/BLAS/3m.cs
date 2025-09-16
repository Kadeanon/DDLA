using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    public static void GeMM
        (TransType aTrans, TransType bTrans,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
    {
        var (m, n) = GetLengths(C);
        if (m == 0 || n == 0) return;
        var (m2, k) = GetLengthsAfterTrans(A, aTrans);
        if (k == 0) return;
        if (m2 != m) throw new ArgumentException($"Dimensions of matrix A must be match!");
        CheckLengthsAfterTrans(B, bTrans, k, n);

        Source.GeMM(aTrans, bTrans,
            m, n, k,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void GeMM
        (scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
        => GeMM(TransType.NoTrans,
            TransType.NoTrans,
            alpha, A, B, beta, C);

    public static void GeMMt
        (UpLo cUplo, TransType aTrans, 
        TransType bTrans, 
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
    {
        var m = CheckSymmMatLength(C, cUplo);
        if (m == 0) return;
        var (ma, k) = GetLengthsAfterTrans(A, aTrans);
        if (k == 0) return;
        CheckLengthsAfterTrans(B, bTrans, k, m);
        if (ma != m)
            throw new ArgumentException($"Dimensions of matrix A must be match!");

        Source.GeMMt(cUplo,
            aTrans,
            bTrans,
            m, k,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void GeMMt
        (UpLo cUplo, 
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
        => GeMMt(cUplo, TransType.NoTrans, 
            TransType.NoTrans,
            alpha, A, B, beta, C);

    public static void SyMM
        (SideType aSide, UpLo aUplo, 
        TransType bTrans,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
    {
        var (m, n) = GetLengths(C);
        if (m == 0 || n == 0) return;
        CheckLengthsAfterTrans(B, bTrans, m, n);
        var aExp = aSide == SideType.Left ? m : n;
        var aLength = CheckSymmMatLength(A, aUplo);
        if (aExp != aLength)
            throw new ArgumentException($"Dimensions of matrixs must be match!");
        Source.SyMM(aSide, aUplo, ConjType.NoConj, bTrans,
            m, n,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void SyMM
        (SideType aSide, UpLo aUplo,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
        => SyMM(aSide, aUplo, TransType.NoTrans,
            alpha, A, B, beta, C);

    public static void SyRk
        (UpLo cUplo, TransType aTrans,
        scalar alpha,
        in matrix A,
        scalar beta,
        in matrix C)
    {
        var m = CheckSymmMatLength(C, cUplo);
        if (m == 0) return;
        var (ma, k) = GetLengthsAfterTrans(A, aTrans);
        if (k == 0) return;
        if (ma != m)
            throw new ArgumentException("Dimensions of matrixs A must be match!");
        Source.SyRk(cUplo,
            aTrans,
            m, k,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void SyRk
        (UpLo cUplo,
        scalar alpha,
        in matrix A,
        scalar beta,
        in matrix C)
        => SyRk(cUplo, TransType.NoTrans,
            alpha, A, beta, C);

    public static void SyR2k
        (UpLo cUplo, TransType aTrans, 
        TransType bTrans,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
    {
        var m = CheckSymmMatLength(C, cUplo);
        if (m == 0) return;
        var (ma, k) = GetLengthsAfterTrans(A, aTrans);
        if (k == 0) return;
        CheckLengthsAfterTrans(B, bTrans, m, k);
        if (ma != m)
            throw new ArgumentException("Dimensions of matrixs A and B must be match!");
        Source.SyR2k(cUplo, aTrans, bTrans,
            m, k,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void SyR2k
        (UpLo cUplo,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
        => SyR2k(cUplo, TransType.NoTrans, 
            TransType.NoTrans,
            alpha, A, B, beta, C);

    public static void TrMM
        (SideType sidea, UpLo aUplo, 
        TransType aTrans, DiagType aDiag,
        in scalar alpha,
        in matrix A,
        in matrix B)
    {
        var (m, n) = GetLengths(B);
        if (m == 0 || n == 0) return;
        var aLength = CheckSymmMatLength(A, aUplo);
        var aExpected = sidea == SideType.Left ? m : n;
        if (aLength != aExpected)
            throw new ArgumentException("Dimensions of matrixs A must be match!");

        Source.TrMM(sidea, aUplo, aTrans, aDiag,
            m, n,
            alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    public static void TrMM
        (SideType sidea, UpLo aUplo,
        in scalar alpha,
        in matrix A,
        in matrix B)
        => TrMM(sidea, aUplo,
            TransType.NoTrans, DiagType.NonUnit,
            alpha, A, B);

    public static void TrMM3
        (SideType sidea, UpLo aUplo,
        TransType aTrans, DiagType aDiag,
        TransType bTrans,
        in scalar alpha,
        in matrix A,
        in matrix B,
        in scalar beta,
        in matrix C)
    {
        var (m, n) = GetLengths(C);
        if (m == 0 || n == 0) return;
        CheckLengthsAfterTrans(B, bTrans, m, n);
        var aLength = CheckSymmMatLength(A, aUplo);
        var aExpected = sidea == SideType.Left ? m : n;
        if (aLength != aExpected)
            throw new ArgumentException("Dimensions of matrixs A must be match!");

        Source.TrMM3(sidea, aUplo, aTrans, aDiag, bTrans,
            m, n,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void TrMM3
        (SideType sidea, UpLo aUplo,
        in scalar alpha,
        in matrix A,
        in matrix B,
        in scalar beta,
        in matrix C)
        => TrMM3(sidea, aUplo,
            TransType.NoTrans, DiagType.NonUnit,
            TransType.NoTrans,
            alpha, A, B, beta, C);

    /// <summary>
    /// If <paramref name="aSide"/> is <see cref="SideType.Left"/>,
    /// solve Trans(<paramref name="A"/>) * X = alpha * <paramref name="B"/>, 
    /// and overwrite <paramref name="B"/> with X.
    /// <br />
    /// Or If <paramref name="aSide"/> is <see cref="SideType.Right"/>,
    /// solve X * Trans(<paramref name="A"/>) = alpha * <paramref name="B"/>, 
    /// and overwrite <paramref name="B"/> with X, 
    /// </summary>
    /// <exception cref="ArgumentException"></exception>

    public static void TrSM
        (SideType aSide, UpLo aUplo,
        TransType aTrans, DiagType aDiag,
        in scalar alpha,
        in matrix A,
        in matrix B)
    {
        var (m, n) = GetLengths(B);
        if (m == 0 || n == 0) return;
        var aLength = CheckSymmMatLength(A, aUplo);
        var aExpected = aSide == SideType.Left ? m : n;
        if (ShouldCheck && aLength != aExpected)
            throw new ArgumentException("Dimensions of matrixs A must be match!");
        Source.TrSM(aSide, aUplo, aTrans, aDiag,
            B.Rows, B.Cols,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    public static void TrSM
        (SideType aSide, UpLo aUplo,
        in scalar alpha,
        in matrix A,
        in matrix B)
        => TrSM(aSide, aUplo, 
            TransType.NoTrans, DiagType.NonUnit,
            alpha, A, B);
}
