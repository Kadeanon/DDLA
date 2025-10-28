using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;
using DDLA.Einsum;
using DDLA.Utilities;
using DDLA.Misc.Pools;

namespace DDLA.BLAS.Managed;

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
        var AEffective = aTrans.HasFlag(TransType.OnlyTrans) ?
            A.T : A;
        var BEffective = bTrans.HasFlag(TransType.OnlyTrans) ?
            B.T : B;
        var (m, n) = GetLengths(C);
        var (m2, k) = GetLengths(AEffective);
        if (m2 != m) throw new ArgumentException($"Dimensions of matrix A must be match!");

        if (m == 0 || n == 0) return;
        CheckLengths(BEffective, k, n);

        Scal(beta, C);
        if (k == 0) return;

        GeMMInner(m, n, k,
            alpha,
            AEffective, BEffective,
            beta, C);
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

    public static void GeMMT
        (UpLo cUplo, TransType aTrans,
        TransType bTrans,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
    {
        var AEffective = aTrans.HasFlag(TransType.OnlyTrans) ?
            A.T : A;
        var BEffective = bTrans.HasFlag(TransType.OnlyTrans) ?
            B.T : B;
        var (m, n) = GetLengths(C);
        var (m2, k) = GetLengths(AEffective);
        if (m2 != m) throw new ArgumentException(
            $"Dimensions of matrix A must be match!");

        if (m == 0 || n == 0) return;
        CheckLengths(BEffective, k, n);

        Scal(cUplo, beta, C);
        if (k == 0) return;

        var kernel = new GEMMKernel();
        var MC = kernel.mc;
        var NC = kernel.nc;
        var KC = kernel.kc;
        var MR = kernel.mr;
        var NR = kernel.nr;

        var CEffective = C;
        var shouldTrans =
            (kernel.preferCol && C.RowStride < C.ColStride) ||
            (!kernel.preferCol && C.ColStride < C.RowStride);
        if (shouldTrans)
        {
            CEffective = C.T;
            cUplo = Transpose(cUplo);
            (m, n) = (n, m);
            (AEffective, BEffective) = (BEffective.T, AEffective.T);
        }
        var diagOrig = CEffective.DiagOffset;

        var MCEffective = Math.Min(MC, m.Align(MR));
        var NCEffective = Math.Min(NC, n.Align(NR));
        using var aBufferHandler = InternelPool.TakeArraySegement(
            MCEffective * KC, out var bufferA, init: false);
        using var bBufferHandler = InternelPool.TakeArraySegement(
            KC * NCEffective, out var bufferB, init: false);

        for (int ic = 0; ic < n; ic += MC)
        {
            var mc = Math.Min(MC, (m - ic));
            for (int pc = 0; pc < k; pc += KC)
            {
                var kc = Math.Min(KC, k - pc);
                
                var bufferAEffective = bufferA.Slice(0, mc.Align(MR) * kc);
                GEMMPack(AEffective, ic, mc, pc, kc, bufferAEffective, MR);

                for (int jc = 0; jc < n; jc += NC)
                {
                    var nc = Math.Min(NC, n - jc);
                    var diag = diagOrig + ic - jc;
                    
                    var bufferBEffective = bufferB.Slice(0, kc * nc.Align(NR));
                    GEMMPack(BEffective, pc, kc, jc, nc, bufferBEffective, NR, packB: true);
                    var subC = CEffective.SliceSubUncheck(ic, mc, jc, nc);
                    var macroKernel = new GEMMTMacroKernel(MR, NR,
                        kc, cUplo, diag,
                        alpha,
                        bufferAEffective,
                        bufferBEffective,
                        subC,
                        kernel);
                    macroKernel.Invoke();
                }
            }
        }
    }

    public static void GeMMT
        (UpLo cUplo,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
        => GeMMT(cUplo, TransType.NoTrans,
            TransType.NoTrans,
            alpha, A, B, beta, C);

    public static void SyMM
        (SideType aSide, UpLo aUplo,
        TransType aTrans, TransType bTrans,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
    {
        var (m, n) = GetLengths(C);

        var AEffective = A;
        var BEffective = B;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            AEffective = A.T;
            aUplo = Transpose(aUplo);
        }
        if (bTrans.HasFlag(TransType.OnlyTrans))
        {
            BEffective = B.T;
        }
        var k = m;
        var bUplo = UpLo.Dense;
        if (aSide is SideType.Right)
        {
            (AEffective, BEffective) = (B, A);
            k = n;
            bUplo = aUplo;
            aUplo = UpLo.Dense;
        }
        CheckLengths(AEffective, m, k);
        CheckLengths(BEffective, k, n);

        if (m == 0 || n == 0) return;
        CheckLengths(BEffective, k, n);

        Scal(beta, C);
        if (k == 0) return;

        SyMMInner(m, n, k, alpha,
            aUplo, AEffective,
            bUplo, BEffective,
            beta, C);
    }

    public static void SyMM
        (SideType aSide, UpLo aUplo,
        scalar alpha,
        in matrix A,
        in matrix B,
        scalar beta,
        in matrix C)
        => SyMM(aSide, aUplo, TransType.NoTrans, TransType.NoTrans,
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

    public static void TrMM
        (SideType aSide, UpLo aUplo,
        TransType aTrans, DiagType aDiag,
        in scalar alpha,
        in matrix A,
        in matrix B)
    {
        var (m, n) = GetLengths(B);
        if (m == 0 || n == 0) return;
        var aLength = CheckSymmMatLength(A, aUplo);
        var k = aSide == SideType.Left ? m : n;
        if (aLength != k)
            throw new ArgumentException("Dimensions of matrixs A must be match!");

        var bUplo = UpLo.Dense;
        var AEffective = A;
        var BEffective = B;
        if (aSide is SideType.Right)
        {
            (AEffective, BEffective) = (B, A);
            bUplo = aUplo;
            aUplo = UpLo.Dense;
        }
        TrMMInner(aDiag,
            m, n,
            alpha,
            aUplo, AEffective,
            bUplo, BEffective, B);
    }

    public static void TrMM
        (SideType aSide, UpLo aUplo, 
        DiagType aDiag,
        in scalar alpha, 
        in matrix A,
        in matrix B)
        => TrMM(aSide, aUplo,
            TransType.NoTrans, aDiag,
            alpha, A, B);

    public static void TrMM
        (SideType aSide, UpLo aUplo,
        in scalar alpha,
        in matrix A,
        in matrix B)
        => TrMM(aSide, aUplo,
            TransType.NoTrans, DiagType.NonUnit,
            alpha, A, B);

    public static void TrMM3
        (SideType aSide, UpLo aUplo,
        TransType aTrans, DiagType aDiag,
        TransType bTrans,
        in scalar alpha,
        in matrix A,
        in matrix B,
        in scalar beta,
        in matrix C)
    {
        var (m, n) = GetLengths(C);

        var AEffective = A;
        var BEffective = B;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            AEffective = A.T;
            aUplo = Transpose(aUplo);
        }
        if (bTrans.HasFlag(TransType.OnlyTrans))
        {
            BEffective = B.T;
        }
        var k = m;
        var bUplo = UpLo.Dense;
        var bDiag = DiagType.NonUnit;
        if (aSide is SideType.Right)
        {
            (AEffective, BEffective) = (B, A);
            k = n;
            bUplo = aUplo;
            bDiag = aDiag;
            aUplo = UpLo.Dense;
            aDiag = DiagType.NonUnit;
        }
        CheckLengths(AEffective, m, k);
        CheckLengths(BEffective, k, n);

        if (m == 0 || n == 0) return;
        CheckLengths(BEffective, k, n);

        Scal(beta, C);
        if (k == 0) return;

        TrMM3Inner(m, n, k, alpha,
            aUplo, aDiag, AEffective,
            bUplo, bDiag, BEffective,
            beta, C);
    }

    public static void TrMM3Source
        (SideType aSide, UpLo aUplo,
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
        var aExpected = aSide == SideType.Left ? m : n;
        if (aLength != aExpected)
            throw new ArgumentException("Dimensions of matrixs A must be match!");

        Source.TrMM3(aSide, aUplo, aTrans, aDiag, bTrans,
            m, n,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride,
            in beta,
            ref C.GetHeadRef(), C.RowStride, C.ColStride);
    }

    public static void TrMM3
        (SideType aSide, UpLo aUplo, DiagType aDiag,
        in scalar alpha,
        in matrix A,
        in matrix B,
        in scalar beta,
        in matrix C)
        => TrMM3(aSide, aUplo,
            TransType.NoTrans, aDiag,
            TransType.NoTrans,
            alpha, A, B, beta, C);

    public static void TrMM3
        (SideType aSide, UpLo aUplo,
        in scalar alpha,
        in matrix A,
        in matrix B,
        in scalar beta,
        in matrix C)
        => TrMM3(aSide, aUplo,
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
        var k = aSide == SideType.Left ? m : n;
        if (aLength != k)
            throw new ArgumentException("Dimensions of matrixs A must be match!");

        Scal(alpha, B);

        var bUplo = UpLo.Dense;
        var AEffective = A;
        var BEffective = B;
        if (aSide is SideType.Right)
        {
            (AEffective, BEffective) = (B, A);
            bUplo = aUplo;
            aUplo = UpLo.Dense;
        }
        TrSMInner(aDiag,
            m, n,
            aUplo, AEffective,
            bUplo, BEffective, B);
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
