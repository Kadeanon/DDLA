using CommunityToolkit.HighPerformance;
using CommunityToolkit.HighPerformance.Helpers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace DDLA.Einsum;

internal readonly struct KernelParallel(BSMBlock target, double alpha,
double[] blockA, double[] blockB, int ic, int qc, int jc, GEMMKernel kernel) : IAction
{
    readonly double alpha = alpha;
    readonly GEMMKernel kernel = kernel;
    readonly BSMBlock target = target;
    readonly double[] blockA = blockA, blockB = blockB;
    readonly int ic = ic, qc = qc, jc = jc;

    public void Invoke()
    {
        int mr = kernel.mr;
        int iCycles = (ic - 1) / mr + 1;
        ParallelHelper.For(0, iCycles, this, 4);
    }

    public unsafe void Invoke(int iCycle)
    {
        int mr = kernel.mr, nr = kernel.nr;
        Span<double> buffer = stackalloc double[mr * nr];
        int ii = iCycle * mr;
        var blockASpan = blockA.AsSpan(ii * qc, mr * qc);
        var ir = Math.Min(mr, ic - ii);
        var blockBSpan = blockB.AsSpan();
        if (Sse.IsSupported && (iCycle + 1) * mr < ic)
        {
            Sse.Prefetch2(Unsafe.AsPointer(
                ref blockASpan.DangerousGetReferenceAt(mr * qc)));
        }
        for (int jj = 0; jj < jc; jj += nr)
        {
            var currentBSpan = blockB.AsSpan(jj * qc, nr * qc);
            if (Sse.IsSupported && jj < jc - nr)
            {
                Sse.Prefetch1(Unsafe.AsPointer(
                    ref currentBSpan.DangerousGetReferenceAt(nr * qc)));
            }
            var jr = Math.Min(nr, jc - jj);
            target.Prefetch(ii, ir, jj, jr, kernel.preferCol);
            buffer.Clear();
            kernel.Kernel(blockASpan, currentBSpan, buffer, mr, nr, qc);
            target.UnpackAxpy(alpha, ii, ir, jj, jr, buffer, kernel.preferCol);
        }
    }
}