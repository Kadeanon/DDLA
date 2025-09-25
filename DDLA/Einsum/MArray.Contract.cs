using DDLA.Einsum;
using DDLA.Einsum.EinsumTree;
using DDLA.Misc;
using DDLA.Utilities;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DDLA.Core;

public partial class MArray
{
    public static int DegreeOfParallelism
    {
        get => degreeOfParallelism;
        set => degreeOfParallelism =
            Math.Clamp(value, 1, Environment.ProcessorCount);
    }


    private static int degreeOfParallelism
        = Environment.ProcessorCount / 2;

    public static ArrayPool<int> ContractNintPool { get; } = ArrayPool<int>.Create();


    #region BSMTC

    public static void Contract(double beta, MArray result,
        string expression, double alpha, MArray left, MArray right)
    {
        Contract(result.ScaledBy(beta), expression, alpha, left, right);
    }

    public static void Contract(double beta, MArray result,
        string expression, MArray left, MArray right)
    {
        Contract(result.ScaledBy(beta), expression, 1.0, left, right);
    }

    public static void Contract(MArray result, string expression,
        MArray left, MArray right)
        => Contract(result, expression, 1.0, left, right);

    public static void Contract(MArray result, string expression,
        double alpha, MArray left, MArray right)
    {
        if (alpha == 0.0)
            return;
        var symbols = expression.Split("->", StringSplitOptions.TrimEntries);
        if (symbols.Length != 2)
            throw new ArgumentException("Expression must contain exactly one '->' symbol.");
        var resultSymbol = symbols[1];
        symbols = symbols[0].Split(",", StringSplitOptions.TrimEntries);
        if (symbols.Length != 2)
            throw new ArgumentException("Expression must contain exactly two symbols before '->'.");
        var leftSymbol = symbols[0];
        var rightSymbol = symbols[1];

        var indicesA = left.Diagonal(leftSymbol);
        var indicesB = right.Diagonal(rightSymbol);
        var indicesC = result.Diagonal(resultSymbol);
        IndiceUtils.Divide(indicesA, indicesB, indicesC,
            out var indicesAB, out var indicesAC, out var indicesBC, out var indicesABC);
        Span<TripleIndice> offsets = stackalloc TripleIndice[
            indicesA.Count + indicesB.Count + indicesC.Count + indicesABC.Count];
        int currentIndex = 0;
        foreach (var indice in indicesABC.Values.OrderBy(ind => -ind.CStride))
        {
            offsets[currentIndex++] = indice;
        }
        foreach (var indice in indicesC.Values.OrderBy(ind => -ind.Stride))
        {
            offsets[currentIndex++] = new TripleIndice(
                indice.Length, 0, 0, indice.Stride);
        }
        foreach (var indice in indicesA.Values.OrderBy(ind => -ind.Stride))
        {
            offsets[currentIndex++] = new TripleIndice(
                indice.Length, indice.Stride, 0, 0);
        }
        foreach (var indice in indicesB.Values.OrderBy(ind => -ind.Stride))
        {
            offsets[currentIndex++] = new TripleIndice(
                indice.Length, 0, indice.Stride, 0);
        }

        indicesAC.Fold();
        var indicesM = indicesAC.Select(x => x.Value).ToArray();
        Array.Sort(indicesM, (x, y) => y.BStride.CompareTo(x.BStride));
        indicesBC.Fold();
        var indicesN = indicesBC.Select(x => x.Value).ToArray();
        Array.Sort(indicesN, (x, y) => y.BStride.CompareTo(x.BStride));
        indicesAB.Fold();
        var indicesK = indicesAB.Select(x => x.Value).ToArray();
        Array.Sort(indicesK, (x, y) => y.AStride.CompareTo(x.AStride));

        if (indicesM.Length == 0)
            indicesM = [new(1, 1, 1)];
        if (indicesN.Length == 0)
            indicesN = [new(1, 1, 1)];
        if (indicesK.Length == 0)
            indicesK = [new(1, 1, 1)];

        if (indicesN.Last().BStride == 1)
        {
            (left, right) = (right, left);
            (indicesM, indicesN) = (indicesN, indicesM);
            foreach (ref var indice in indicesK.AsSpan())
            {
                indice = indice.Swap();
            }
            foreach (ref var indice in offsets)
            {
                indice =
                    new(indice.Length, indice.BStride, indice.AStride, indice.CStride);
            }
        }

        BlockScatterContract_Silent
            (offsets, alpha, left, 0, right, 0, result, 0, indicesM, indicesN, indicesK);
    }

    public static MArray Contract(string expression, MArray left,
        MArray right)
        => Contract(expression, 1.0, left, right);

    public static MArray Contract(string expression,
        double alpha, MArray left, MArray right)
    {
        var symbols = expression.Split("->",
            StringSplitOptions.TrimEntries);
        if (symbols.Length != 2)
            throw new ArgumentException(
                "Expression must contain exactly one '->' symbol.");
        var resultSymbol = symbols[1];
        symbols = symbols[0].Split(",",
            StringSplitOptions.TrimEntries);
        if (symbols.Length != 2)
            throw new ArgumentException(
                "Expression must contain exactly two symbols before '->'.");
        var leftSymbol = symbols[0];
        var rightSymbol = symbols[1];

        var indicesA = left.Diagonal(leftSymbol);
        var indicesB = right.Diagonal(rightSymbol);
        Span<int> lengthsC = stackalloc int[resultSymbol.Length];
        for (int i = resultSymbol.Length - 1; i >= 0; i--)
        {
            int stride = 1;
            char symbol = resultSymbol[i];
            if (indicesA.TryGetValue(symbol, out var input))
            {
                int length = input.Length;
                lengthsC[i] = length;
                stride *= length;
            }
            else if (indicesB.TryGetValue(symbol, out input))
            {
                int length = input.Length;
                lengthsC[i] = length;
                stride *= length;
            }
            else
            {
                throw new ArgumentException(
                    $"Symbol '{symbol}' not found in either tensor.");
            }
        }
        MArray result = Create(lengthsC);
        var indicesC = result.Diagonal(resultSymbol);
        indicesA = indicesA.Where(kvp => kvp.Value.Length > 1).
            ToDictionary();
        indicesB = indicesB.Where(kvp => kvp.Value.Length > 1).
            ToDictionary();
        indicesC = indicesC.Where(kvp => kvp.Value.Length > 1).
            ToDictionary();
        IndiceUtils.Divide(indicesA, indicesB, indicesC,
            out var indicesAB, out var indicesAC, out var indicesBC,
            out var indicesABC);
        Span<TripleIndice> offsets = stackalloc TripleIndice[
            indicesA.Count + indicesB.Count + indicesC.Count
            + indicesABC.Count];
        int currentIndex = 0;
        foreach (var indice in indicesABC.Values.OrderBy(
            ind => -ind.CStride))
        {
            offsets[currentIndex++] = indice;
        }
        foreach (var indice in indicesC.Values.OrderBy(
            ind => -ind.Stride))
        {
            offsets[currentIndex++] = new TripleIndice(
                indice.Length, 0, 0, indice.Stride);
        }
        foreach (var indice in indicesA.Values.OrderBy(
            ind => -ind.Stride))
        {
            offsets[currentIndex++] = new TripleIndice(
                indice.Length, indice.Stride, 0, 0);
        }
        foreach (var indice in indicesB.Values.OrderBy(
            ind => -ind.Stride))
        {
            offsets[currentIndex++] = new TripleIndice(
                indice.Length, 0, indice.Stride, 0);
        }

        indicesAC.Fold();
        var indicesM = indicesAC.Select(x => x.Value).ToArray();
        Array.Sort(indicesM, (x, y) => y.BStride.CompareTo(x.BStride));
        indicesBC.Fold();
        var indicesN = indicesBC.Select(x => x.Value).ToArray();
        Array.Sort(indicesN, (x, y) => y.BStride.CompareTo(x.BStride));
        indicesAB.Fold();
        var indicesK = indicesAB.Select(x => x.Value).ToArray();
        Array.Sort(indicesK, (x, y) => y.AStride.CompareTo(x.AStride));

        if (indicesM.Length == 0)
            indicesM = [new DoubleIndice(1, 1, 1)];
        if (indicesN.Length == 0)
            indicesN = [new DoubleIndice(1, 1, 1)];
        if (indicesK.Length == 0)
            indicesK = [new DoubleIndice(1, 1, 1)];

        if (indicesN.Last().BStride == 1)
        {
            (left, right) = (right, left);
            (indicesM, indicesN) = (indicesN, indicesM);
            foreach (ref var indice in indicesK.AsSpan())
            {
                indice = indice.Swap();
            }
            foreach (ref var indice in offsets)
            {
                indice =
                    new(indice.Length,
                    indice.BStride, indice.AStride, indice.CStride);
            }
        }

        BlockScatterContract_Silent
            (offsets, alpha, left, 0, right, 0, result, 0,
            indicesM, indicesN, indicesK);

        return result;
    }

    private static void BlockScatterContract(double alpha,
        MArray left, MArray right, MArray result,
        DoubleIndice[] indicesM, DoubleIndice[] indicesN,
        DoubleIndice[] indicesK)
    {
        GEMMKernel kernel = default;
        var indicesMA = indicesM.Select(x => x.A).ToArray();
        var indicesKA = indicesK.Select(x => x.A).ToArray();
        var indicesKB = indicesK.Select(x => x.B).ToArray();
        var indicesNB = indicesN.Select(x => x.A).ToArray();
        var indicesMC = indicesM.Select(x => x.B).ToArray();
        var indicesNC = indicesN.Select(x => x.B).ToArray();
        int mc = kernel.mc, nc = kernel.nc, kc = kernel.kc;
        int mr = kernel.mr, nr = kernel.nr, kr = kernel.kr;

        using var matrixA = new BlockScatterMatrix(left, 0,
            indicesMA, mr, indicesKA, kr);
        using var matrixB = new BlockScatterMatrix(right, 0,
            indicesKB, kr, indicesNB, nr);
        using var matrixC = new BlockScatterMatrix(result, 0,
            indicesMC, mr, indicesNC, nr);
        //matrixB.Transpose();

        var m = matrixA.rowLength;
        var n = matrixB.colLength;
        var k = matrixA.colLength;

        int mMax = (int)Math.Min(mc, m).Align(mr);
        int nMax = (int)Math.Min(nc, n).Align(nr);
        int kMax = (int)Math.Min(kc, k).Align(kr);
        var blockA = ArrayPool<double>.Shared.Rent(kMax * mMax);
        var blockB = ArrayPool<double>.Shared.Rent(kMax * nMax);

        for (int i = 0; i < m; i += mc)
        {
            int ic = (int)Math.Min(mc, m - i);
            for (int q = 0; q < k; q += kc)
            {
                int qc = (int)Math.Min(kc, k - q);
                int qc_align = qc.Align(kr);
                var sourceA = matrixA.Slice(i, ic, q, qc);
                sourceA.Pack(blockA);
                for (int j = 0; j < n; j += nc)
                {
                    int jc = (int)Math.Min(nc, n - j);
                    var sourceB = matrixB.Slice(q, qc, j, jc,
                        trans: true);
                    sourceB.Pack(blockB);
                    var targetC = matrixC.Slice(i, ic, j, jc);
                    KernelParallel kernelParallel =
                        new(targetC, alpha, blockA, blockB,
                        ic, qc_align, jc, kernel);
                    kernelParallel.Invoke();
                }
            }
        }

        ArrayPool<double>.Shared.Return(blockA);
        ArrayPool<double>.Shared.Return(blockB);
    }

    private static void BlockScatterContract_Silent
        (ReadOnlySpan<TripleIndice> batchs, double alpha,
        MArray left, int leftOffset,
        MArray right, int rightOffset,
        MArray result, int resultOffset,
        DoubleIndice[] indicesM, DoubleIndice[] indicesN,
        DoubleIndice[] indicesK)
    {
        if (batchs.Length > 1)
        {
            var currentBatch = batchs[0];
            batchs = batchs[1..];
            for (int i = 0; i < currentBatch.Length; i++)
            {
                BlockScatterContract_Silent
                    (batchs, alpha, left, leftOffset,
                    right, rightOffset,
                    result, resultOffset,
                    indicesM, indicesN, indicesK);
                leftOffset += currentBatch.AStride;
                rightOffset += currentBatch.BStride;
                resultOffset += currentBatch.CStride;
            }
        }
        else
        {
            GEMMKernel kernel = new();
            var indicesMA = indicesM.Select(x => x.A).ToArray();
            var indicesKA = indicesK.Select(x => x.A).ToArray();
            var indicesKB = indicesK.Select(x => x.B).ToArray();
            var indicesNB = indicesN.Select(x => x.A).ToArray();
            var indicesMC = indicesM.Select(x => x.B).ToArray();
            var indicesNC = indicesN.Select(x => x.B).ToArray();
            int mc = kernel.mc, nc = kernel.nc, kc = kernel.kc;
            int mr = kernel.mr, nr = kernel.nr, kr = kernel.kr;

            var m = indicesMA.AsSpan().TotalLength();
            var n = indicesNB.AsSpan().TotalLength();
            var k = indicesKA.AsSpan().TotalLength();

            int mMax = (int)Math.Min(mc, m).Align(mr);
            int nMax = (int)Math.Min(nc, n).Align(nr);
            int kMax = (int)Math.Min(kc, k).Align(kr);
            int qAlign = k.Align(kr);
            int nAlign = n.Align(nr);
            double[] blockA = ArrayPool<double>.Shared.Rent(
                kMax * mMax);
            double[] blockB = ArrayPool<double>.Shared.Rent(
                kMax * nMax);

            using var matrixA = new BlockScatterMatrix(left,
                leftOffset, indicesMA, mr, indicesKA, kr);
            using var matrixB = new BlockScatterMatrix(right,
                rightOffset, indicesKB, kr, indicesNB, nr);
            using var matrixC = new BlockScatterMatrix(result,
                resultOffset, indicesMC, mr, indicesNC, nr);

            var currentBatch = batchs.Length == 1 ?
                batchs[0] : new(1, 0, 0, 0);
            for (int iBatch = 0; iBatch < currentBatch.Length; iBatch++)
            {
                for (int i = 0; i < m; i += mc)
                {
                    int ic = (int)Math.Min(mc, m - i);
                    for (int q = 0; q < k; q += kc)
                    {
                        int qc = (int)Math.Min(kc, k - q);
                        int qc_align = qc.Align(kr);
                        var sourceA = matrixA.Slice(i, ic, q, qc);
                        sourceA.Pack(blockA);
                        for (int j = 0; j < n; j += nc)
                        {
                            int jc = (int)Math.Min(nc, n - j);
                            var sourceB = matrixB.Slice(q, qc, j, jc,
                                trans: true);
                            sourceB.Pack(blockB);
                            var targetC = matrixC.Slice(i, ic, j, jc);
                            KernelParallel kernelParallel =
                                new(targetC, alpha, blockA, blockB,
                                ic, qc_align, jc, kernel);
                            kernelParallel.Invoke();
                        }
                    }
                }
                matrixA.AddOffset(currentBatch.AStride);
                matrixB.AddOffset(currentBatch.BStride);
                matrixC.AddOffset(currentBatch.CStride);
            }

            ArrayPool<double>.Shared.Return(blockA);
            ArrayPool<double>.Shared.Return(blockB);
        }
    }

    #endregion

    public static void Einsum(MArray result, string expression,
        params ReadOnlySpan<MArray> inputs)
        => result.AddedBy(Einsum(expression, inputs));

    public static void Einsum(MArray result, string expression,
        double alpha, params ReadOnlySpan<MArray> inputs)
        => result.AddedByScaled(alpha, Einsum(expression, inputs));

    public static MArray Einsum(string expression,
        double alpha, params ReadOnlySpan<MArray> inputs)
        => Einsum(expression, inputs).ScaledBy(alpha);

    public static MArray Einsum(string expression,
        params ReadOnlySpan<MArray> inputs)
    {
        var state = new EinsumState(expression, inputs);
        var ir = state.Parse();
        var result = ir.Invoke(inputs);
        return result;
    }

    internal static int GetLevel(DoubleIndice[] indicesM, DoubleIndice[] indicesN,
        DoubleIndice[] indicesK, double kernelCost)
    {
        double cost = GetCost(
            indicesM, indicesN, indicesK, kernelCost);
        if (cost < 1.25)
            return 1;
        else if (cost < 2)
            return 2;
        else if (cost < 10)
            return 3;
        else
            return 4;
    }

    internal static double GetCost(DoubleIndice[] indicesM, DoubleIndice[] indicesN,
        DoubleIndice[] indicesK, double kernelCost)
    {
        var indiceM = indicesM.Last();
        var indiceN = indicesN.Last();
        var indiceK = indicesK.Last();
        int SizeM = indicesM.AsSpan().TotalLength();
        int SizeN = indicesN.AsSpan().TotalLength();
        int SizeK = indicesK.AsSpan().TotalLength();

        return GetCost(indiceM, indiceN, indiceK,
            SizeM, SizeN, SizeK, kernelCost);
    }

    internal static double GetCost(DoubleIndice indiceM, DoubleIndice indiceN,
        DoubleIndice indiceK, int sizeM, int sizeN, int sizeK, double kernelCost)
    {
        GEMMKernel kernel = new();
        double cost = 0;
        int mc = kernel.mc;
        int mr = kernel.mr;
        int nc = kernel.nc;
        int nr = kernel.nr;
        int kc = kernel.kc;
        int strideAM = indiceM.AStride;
        int strideAK = indiceK.AStride;
        int strideBN = indiceN.AStride;
        int strideBK = indiceK.BStride;
        int strideCM = indiceM.BStride;
        int strideCN = indiceN.BStride;

        double packA = GetCostPerAccess
            (strideAM, strideAK, mc, kc);
        packA /= sizeN;

        double packB = GetCostPerAccess
            (strideBN, strideBK, nc, kc);
        packB *= GetCycles(sizeM, mc);
        packB /= sizeM;

        double unpackC = GetCostPerAccess
            (strideCM, strideCN, mc, nc,
            prefetchRow: true);
        unpackC *= GetCycles(sizeK, kc);
        unpackC /= sizeK;

        double x = kernelCost;
        double kernelFactor = 1;
        if (sizeK < kc)
        {
            x = (kc * kernelFactor + sizeK)
                / (kernelFactor + 1) / sizeK;
        }

        double parallelFactorM = 1 +
            (DegreeOfParallelism - 1) * 0.9;
        sizeM = Math.Min(sizeM, mc);
        var iBlockNum = (sizeM - 1) / mr + 1;
        var iBlockBatch = (iBlockNum - 1) / 16 + 1;
        var iCoreNum = Math.Min(iBlockBatch, DegreeOfParallelism);
        parallelFactorM /= iCoreNum;

        double parallelFactorN = 1 +
            (DegreeOfParallelism - 1) * 0.9;
        sizeN = Math.Min(sizeN, nc);
        var jBlockNum = (sizeN - 1) / nr + 1;
        var jBlockBatch = (jBlockNum - 1) / 16 + 1;
        var jCoreNum = Math.Min(jBlockBatch, DegreeOfParallelism);
        parallelFactorN /= jCoreNum;

        packA *= parallelFactorM;
        packB *= parallelFactorN;
        unpackC *= parallelFactorM;
        x *= parallelFactorM;

        cost += packA;
        cost += packB;
        cost += unpackC;
        cost += x;
        cost /= kernelCost + 0.25;
        return cost;

        static double GetCostPerAccess(int rowStride, int colStride,
        int rowBlock, int colBlock, bool prefetchRow = false)
        {
            double factor = 1.0;
            double simpleFactor = factor * 3;
            int TLBSize = 4096 / 8;
            int cacheLineSize = 64 / 8;

            double rowFactor = 1.0;
            double totalAccess = rowStride * rowBlock;
            double totalPages = totalAccess / TLBSize;
            totalPages = Math.Min(rowBlock, totalPages);
            rowFactor *= totalPages / rowBlock * 300;
            double totalLines = totalAccess / cacheLineSize;
            totalLines = Math.Min(rowBlock, totalLines);
            rowFactor += totalLines / rowBlock * 80;
            if (!prefetchRow)
            {
                rowFactor *= 0.5;
            }
            else
            {
                rowFactor *= 0.8;
            }
            factor += rowFactor;

            double colFactor = 1.0;
            totalAccess = colStride * colBlock;
            totalPages = totalAccess / TLBSize;
            totalPages = Math.Min(colBlock, totalPages);
            colFactor *= totalPages / colBlock * 300;
            totalLines = totalAccess / cacheLineSize;
            totalLines = Math.Min(colBlock, totalLines);
            colFactor += totalLines / colBlock * 80;
            colFactor *= 0.8;
            factor += colFactor;

            return Math.Max(simpleFactor, factor);
        }

        static int GetCycles(int length, int block)
        {
            return (length - 1) / block + 1;
        }
    }
}
