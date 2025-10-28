using DDLA.Misc.Flags;
using System.Diagnostics;
using DDLA.Einsum;
using DDLA.Utilities;
using DDLA.Misc.Pools;
using scalar = double;
using matrix = DDLA.Core.MatrixView;

namespace DDLA.BLAS.Managed;

public interface IMacroKernel : IActionEX
{
    public int MR { get; }

    public int NR { get; }

    public int MC { get; }

    public int NC { get; }

    public int KC { get; }
}

public readonly struct GEMMMacroKernel : IMacroKernel
{
    public int MR { get; }

    public int NR { get; }

    public int MC { get; }

    public int NC { get; }

    public int KC { get; }

    public int MCAligned => (MC + MR - 1) / MR * MR;

    public int NCAligned => (NC + NR - 1) / NR * NR;

    public scalar Alpha { get; }

    public ArraySegment<scalar> ABuffer { get; }

    public ArraySegment<scalar> BBuffer { get; }

    public matrix C { get; }

    public GEMMKernel Kernel { get; }

    public GEMMMacroKernel(int mr, int nr, int mc, int nc, int kc,
        scalar alpha, ArraySegment<scalar> aBuffer,
        ArraySegment<scalar> bBuffer, in matrix c, GEMMKernel kernel)
    {
        MR = mr;
        NR = nr;
        MC = mc;
        NC = nc;
        KC = kc;
        Alpha = alpha;
        ABuffer = aBuffer;
        BBuffer = bBuffer;
        C = c;
        Kernel = kernel;
        Debug.Assert(ABuffer.Count >= MCAligned * KC);
        Debug.Assert(BBuffer.Count >= KC * NCAligned);
        Debug.Assert(c.Rows == MC && c.Cols == NC);
    }

    public void Invoke()
    {
        int iCycles = MCAligned / MR;
        ParallelHelperEX.For(0, iCycles, this, 4,
            Environment.ProcessorCount / 2);
    }

    public void Invoke(int i)
    {
        var length = KC * MR;
        var offset = i * length;
        var actualMR = Math.Min(MC - i * MR, MR);
        Span<scalar> bufferA = ABuffer.AsSpan(offset, length);
        using var bufferHandler = InternelPool.TakeMatrix
            (MR, NR, out var bufferCMatrix, !Kernel.preferCol, init: false);
        Span<scalar> bufferC = bufferCMatrix.GetSpan();
        for (int j = 0; j < NCAligned; j += NR)
        {
            var actualNR = Math.Min(NC - j, NR);
            var blockMatrixC = C.SliceSubUncheck(i * MR, actualMR, j, actualNR);
            var blockMatrixBuffer = bufferCMatrix.SliceSubUncheck(0, actualMR,
                0, actualNR);
            Span<scalar> bufferB = BBuffer.AsSpan(j * KC, KC * NR);
            Kernel.Kernel(bufferA, bufferB, bufferC, MR, NR, KC);
            // Add back
            BlasProvider.Axpy(Alpha, blockMatrixBuffer, blockMatrixC);
        }
    }
}

public readonly struct GEMMTMacroKernel : IMacroKernel
{
    public int MR { get; }

    public int NR { get; }

    public int MC => C.Rows;

    public int NC => C.Cols;

    public int KC { get; }

    public int MCAligned => MC.Align(MR);

    public int NCAligned => NC.Align(NR);

    public UpLo UpLo { get; }

    public int DiagOffset { get; }

    public scalar Alpha { get; }

    public ArraySegment<scalar> ABuffer { get; }

    public ArraySegment<scalar> BBuffer { get; }

    public matrix C { get; }

    public GEMMKernel Kernel { get; }

    public GEMMTMacroKernel(int mr, int nr, int kc,
        UpLo uplo, int diag, scalar alpha, ArraySegment<scalar> aBuffer,
        ArraySegment<scalar> bBuffer, in matrix c, GEMMKernel kernel)
    {
        MR = mr;
        NR = nr;
        KC = kc;
        UpLo = uplo;
        DiagOffset = diag;
        Alpha = alpha;
        ABuffer = aBuffer;
        BBuffer = bBuffer;
        C = c;
        Kernel = kernel;
        Debug.Assert(ABuffer.Count >= MCAligned * KC);
        Debug.Assert(BBuffer.Count >= KC * NCAligned);
    }

    public void Invoke()
    {
        int iCycles = MCAligned / MR;
        for (int i = 0; i < iCycles; i++)
        {
            Invoke(i);
        }
        //ParallelHelperEX.For(0, iCycles, this, 4,
        //    Environment.ProcessorCount / 2);
    }

    public void Invoke(int index)
    {
        var i = index * MR;
        var length = KC * MR;
        var offset = index * length;
        var actualMR = Math.Min(MC - i, MR);
        Span<scalar> bufferA = ABuffer.AsSpan(offset, length);
        using var bufferHandler = InternelPool.TakeMatrix
            (MR, NR, out var bufferCMatrix, !Kernel.preferCol, init: false);
        Span<scalar> bufferC = bufferCMatrix.GetSpan();
        for (int j = 0; j < NCAligned; j += NR)
        {
            var diag = DiagOffset + i - j;
            var actualNR = Math.Min(NC - j, NR);
            var blockMatrixC = C.SliceSubUncheck(i, actualMR, j, actualNR);
            var blockMatrixBuffer = bufferCMatrix.SliceSubUncheck(0, actualMR,
                0, actualNR);
            Span<scalar> bufferB = BBuffer.AsSpan(j * KC, KC * NR);

            if (UpLo is UpLo.Upper)
            {
                if (diag >= actualNR)
                    continue;

                Kernel.Kernel(bufferA, bufferB, bufferC, MR, NR, KC);

                // Mask: col < row + diag
                for (int row = 0; row < actualMR; row++)
                {
                    int start = Math.Max(0, Math.Min(actualNR, row + diag));
                    for (int col = start; col < actualNR; col++)
                    {
                        blockMatrixC.AtUncheck(row, col) += Alpha *
                        blockMatrixBuffer.AtUncheck(row, col);
                    }
                }
            }
            else
            {
                if (diag <= -actualMR)
                    continue;

                Kernel.Kernel(bufferA, bufferB, bufferC, MR, NR, KC);

                // Masked: col > row + diag
                for (int row = 0; row < actualMR; row++)
                {
                    int end = Math.Min(Math.Max(0, row + diag + 1), actualNR);
                    for (int col = 0; col < end; col++)
                    {
                        blockMatrixC.AtUncheck(row, col) += Alpha *
                        blockMatrixBuffer.AtUncheck(row, col);
                    }
                }

                // Add back
                // BlasProvider.Axpy(Alpha, blockMatrixBuffer, blockMatrixC);
            }
        }
    }
    public void Invoke2(int index)
    {
        var i = index * MR;
        var diag = DiagOffset + i;
        var length = KC * MR;
        var offset = index * length;
        var actualMR = Math.Min(MC - i, MR);
        Span<scalar> bufferA = ABuffer.AsSpan(offset, length);
        using var bufferHandler = InternelPool.TakeMatrix
            (MR, NR, out var bufferCMatrix, !Kernel.preferCol, init: false);
        Span<scalar> bufferC = bufferCMatrix.GetSpan();
        for (int j = 0; j < NCAligned; j += NR)
        {
            var actualNR = Math.Min(NC - j, NR);
            var blockMatrixC = C.SliceSubUncheck(index * MR, actualMR, j, actualNR);
            var blockMatrixBuffer = bufferCMatrix.SliceSubUncheck(0, actualMR,
                0, actualNR);
            Span<scalar> bufferB = BBuffer.AsSpan(j * KC, KC * NR);

            if (UpLo is UpLo.Upper)
            {
                if (diag >= actualNR) 
                    continue;

                Kernel.Kernel(bufferA, bufferB, bufferC, MR, NR, KC);

                // Masked Copy:  col >= row + diag
                for (int row = 0; row < actualMR; row++)
                {
                    int start = Math.Max(0, row + diag);
                    for (int col = start; col < actualNR; col++)
                    {
                        blockMatrixC.AtUncheck(row, col) += Alpha * 
                            blockMatrixBuffer.AtUncheck(row, col);
                    }
                }
            }
            else
            {
                if (diag <= -actualMR)
                    continue;

                Kernel.Kernel(bufferA, bufferB, bufferC, MR, NR, KC);

                // Masked Copy:  col <= row + diag
                for (int row = 0; row < actualMR; row++)
                {
                    int end = Math.Min(actualNR, row + diag + 1);
                    for (int col = 0; col < end; col++)
                    {
                        blockMatrixC.AtUncheck(row, col) += Alpha * 
                            blockMatrixBuffer.AtUncheck(row, col);
                    }
                }
            }
        }
    }
}
