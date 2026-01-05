using scalar = double;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;
using DDLA.Einsum;
using DDLA.Misc;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
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

        //BLAS.BlasProvider.
        GeMMT(cUplo, aTrans, aTrans, alpha, A, A.T, beta, C);
    }

    public static void SyRk
        (UpLo cUplo,
        scalar alpha,
        in matrix A,
        scalar beta,
        in matrix C)
        => SyRk(cUplo, TransType.NoTrans,
            alpha, A, beta, C);

}
