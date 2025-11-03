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
    private static void TrMM3Inner(int m, int n, int k, scalar alpha,
        UpLo aUplo, DiagType aDiag, matrix A,
        UpLo bUplo, DiagType bDiag, matrix B,
        scalar beta, matrix C)
    {
        var kernel = new GEMMKernel();
        var MC = kernel.mc;
        var NC = kernel.nc;
        var KC = kernel.kc;
        var MR = kernel.mr;
        var NR = kernel.nr;

        var shouldTrans =
            (kernel.preferCol && C.RowStride < C.ColStride) ||
            (!kernel.preferCol && C.ColStride < C.RowStride);
        if (shouldTrans)
        {
            C = C.T;
            (m, n) = (n, m);
            (A, B) = (B.T, A.T);
            (aUplo, bUplo) = (Transpose(bUplo), Transpose(aUplo));
            (aDiag, bDiag) = (bDiag, aDiag);
        }

        var MCEffective = Math.Min(MC, m.Align(MR));
        var NCEffective = Math.Min(NC, n.Align(NR));
        using var aBufferHandler = InternelPool.TakeArraySegement(
            MCEffective * KC, out var bufferA, init: false);
        using var bBufferHandler = InternelPool.TakeArraySegement(
            KC * NCEffective, out var bufferB, init: false);

        for (int ic = 0; ic < m; ic += MC)
        {
            var mc = Math.Min(MC, (m - ic));
            for (int pc = 0; pc < k; pc += KC)
            {
                var kc = Math.Min(KC, k - pc);

                var bufferAEffective = bufferA.Slice(0, mc.Align(MR) * kc);
                TRMMPackA(aDiag, aUplo, ic - pc,
                    A, ic, mc, pc, kc, 
                    bufferAEffective, MR);

                for (int jc = 0; jc < n; jc += NC)
                {
                    var nc = Math.Min(NC, (n - jc));

                    var bufferBEffective = bufferB.Slice(0, kc * nc.Align(NR));
                    TRMMPackB(bDiag, bUplo, pc - jc,
                        B, pc, kc, jc, nc, 
                        bufferBEffective, NR);

                    var subC = C.SliceSubUncheck(ic, mc, jc, nc);
                    var macroKernel = new GEMMMacroKernel(MR, NR,
                        mc, nc, kc,
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
}
