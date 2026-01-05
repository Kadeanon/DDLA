using DDLA.Einsum;
using DDLA.Misc.Flags;
using DDLA.Misc.Pools;
using DDLA.Utilities;
using matrix = DDLA.Core.MatrixView;
using scalar = double;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
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
        var diagOrig = 0;

        var MCEffective = Math.Min(MC, m.Align(MR));
        var NCEffective = Math.Min(NC, n.Align(NR));
        using var aBufferHandler = InternalPool.TakeArraySegement(
            MCEffective * KC, out var bufferA, init: false);
        using var bBufferHandler = InternalPool.TakeArraySegement(
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

}
