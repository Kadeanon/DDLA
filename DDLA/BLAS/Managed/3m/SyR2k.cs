using scalar = double;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;
using DDLA.Einsum;
using DDLA.Misc;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
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

        GeMMT(cUplo, aTrans, bTrans, alpha, A, B.T, beta, C);
        GeMMT(cUplo, bTrans, aTrans, alpha, B, A.T, 1, C);
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

}
