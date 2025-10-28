using CommunityToolkit.HighPerformance;
using DDLA.Core;
using DDLA.Utilities;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DDLA.Einsum;

internal struct PackParallel(in BSMBlock block, Memory<double> buffer) : IActionEX
{
    readonly MArray data = block.data;
    readonly int offset = block.offset;
    readonly Memory<double> buffer = buffer;
    #region Rows
    readonly Memory<int> rowScatters = block.rowScatters;
    public int rowLength = block.rowLength;
    public int rowBlock = block.rowBlock;
    #endregion Rows
    #region Cols
    readonly Memory<int> colScatters = block.colScatters;
    public int colLength = block.colLength;
    public int colBlock = block.colBlock;
    #endregion Cols

    internal readonly void Pack()
    {
        var iBlockNum = (rowLength - 1) / rowBlock + 1;
        ParallelHelperEX.For(
            0, iBlockNum, in this, 4, 4);
    }

    public readonly void Invoke(int i)
    {
        Span<int> zeros = stackalloc int[data.Rank];
        ref double head = ref Unsafe.Add(ref data[zeros], offset);
        int iLengthAlign = rowLength.Align(rowBlock);
        int jLengthAlign = colLength.Align(colBlock);
        var rowScattersBlock = rowScatters.Span.Slice(i * (rowBlock + 1), rowBlock + 1);
        var jBlockNum = (colLength - 1) / colBlock + 1;

        var iBlockSize = Math.Min(rowBlock, rowLength - i * rowBlock);
        var iLast = rowBlock - iBlockSize;
        ref var iToken = ref rowScattersBlock[0];
        ref var iHead = ref Unsafe.Add(ref iToken, 1);
        ref var iBufferHead = ref buffer.Span.DangerousGetReferenceAt(
                    i * jLengthAlign * rowBlock);
        for (int j = 0; j < jBlockNum; j++)
        {
            var jBlockSize = Math.Min(colBlock, colLength - j * colBlock);
            int jLast = colBlock - jBlockSize;
            var colScattersBlock = colScatters.Span.Slice(j * (colBlock + 1), colBlock + 1);
            ref var jToken = ref colScattersBlock[0];
            ref var jHead = ref Unsafe.Add(ref jToken, 1);
            ref var bufferHead = ref Unsafe.Add(ref iBufferHead,
                j * colBlock * rowBlock);
            int jj = 0;
            bool rowCommon = iToken > 0;

            if (jToken > 0)
            {
                if (iToken == 1)
                {
                    ref double origHead = ref Unsafe.Add(ref head, iHead + jHead);
                    for (; jj < jBlockSize; jj++)
                    {
                        MemoryMarshal.CreateReadOnlySpan(ref origHead, iBlockSize).CopyTo
                        (MemoryMarshal.CreateSpan(ref bufferHead, iBlockSize));
                        if (iLast > 0)
                        {
                            ref var last = ref Unsafe.Add(ref bufferHead, iBlockSize);
                            for (int ii = 0; ii < iLast; ii++)
                            {
                                Unsafe.Add(ref last, ii) = 0.0;
                            }
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, rowBlock);
                        origHead = ref Unsafe.Add(ref origHead, jToken);
                    }
                    if (jLast > 0)
                    {
                        MemoryMarshal.CreateSpan(ref bufferHead, jLast * rowBlock)
                            .Clear();
                    }
                }
                else if (iToken > 0)
                {
                    ref double origHead = ref Unsafe.Add(ref head, iHead + jHead);
                    for (; jj < jBlockSize; jj++)
                    {
                        int ii = 0;
                        for (; ii < iBlockSize; ii++)
                        {
                            ref var src = ref Unsafe.Add(
                                ref head,
                                iHead + ii * iToken + jHead + jj * jToken);
                            ref var dst = ref Unsafe.Add(ref bufferHead, ii);
                            dst = src;
                            Unsafe.Add(ref bufferHead, ii) =
                                Unsafe.Add(ref origHead, ii * iToken);
                        }
                        for (; ii < rowBlock; ii++)
                        {
                            Unsafe.Add(ref bufferHead, ii) = 0.0;
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, rowBlock);
                        origHead = ref Unsafe.Add(ref origHead, jToken);
                    }
                    if (jj < colBlock)
                    {
                        MemoryMarshal.CreateSpan(ref bufferHead, (colBlock - jj) * rowBlock)
                            .Clear();
                    }
                }
                else
                {
                    ref double origHead = ref Unsafe.Add(ref head, jHead);
                    for (; jj < jBlockSize; jj++)
                    {
                        int ii = 0;
                        for (; ii < iBlockSize; ii++)
                        {
                            bufferHead = Unsafe.Add(ref origHead,
                                Unsafe.Add(ref iHead, ii));
                            bufferHead = ref Unsafe.Add(ref bufferHead, 1);
                        }
                        for (; ii < rowBlock; ii++)
                        {
                            bufferHead = 0.0;
                            bufferHead = ref Unsafe.Add(ref bufferHead, 1);
                        }
                        origHead = ref Unsafe.Add(ref origHead, jToken);
                    }
                    if (jj < colBlock)
                    {
                        MemoryMarshal.CreateSpan(ref bufferHead, (colBlock - jj) * rowBlock)
                            .Clear();
                    }
                }
            }
            else
            {
                if (iToken == 1)
                {
                    ref var colOffset = ref jHead;
                    ref var origHead = ref Unsafe.Add(ref head, iHead);
                    for (; jj < jBlockSize; jj++)
                    {
                        ref double rowPtr = ref Unsafe.Add(ref origHead, colOffset);
                        MemoryMarshal.CreateReadOnlySpan(ref rowPtr, iBlockSize).CopyTo
                        (MemoryMarshal.CreateSpan(ref bufferHead, iBlockSize));
                        if (iLast > 0)
                        {
                            ref var last = ref Unsafe.Add(ref bufferHead, iBlockSize);
                            for (int ii = 0; ii < iLast; ii++)
                            {
                                Unsafe.Add(ref last, ii) = 0.0;
                            }
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, rowBlock);
                        colOffset = ref Unsafe.Add(ref colOffset, 1);
                    }
                    if (jj < colBlock)
                    {
                        MemoryMarshal.CreateSpan(ref bufferHead, (colBlock - jj) * rowBlock)
                            .Clear();
                    }
                }
                else if (iToken > 0)
                {
                    ref var colOffset = ref jHead;
                    ref var origHead = ref Unsafe.Add(ref head, iHead);
                    for (; jj < jBlockSize; jj++)
                    {
                        ref double rowPtr = ref Unsafe.Add(ref origHead, colOffset);

                        int ii = 0;
                        for (; ii < iBlockSize; ii++)
                        {
                            bufferHead = rowPtr;// Unsafe.Add(ref colPtr, rowOffset);
                            bufferHead = ref Unsafe.Add(ref bufferHead, 1);
                            rowPtr = ref Unsafe.Add(ref rowPtr, iToken);
                        }
                        for (; ii < rowBlock; ii++)
                        {
                            bufferHead = 0.0;
                            bufferHead = ref Unsafe.Add(ref bufferHead, 1);
                        }
                        colOffset = ref Unsafe.Add(ref colOffset, 1);
                    }
                    if (jj < colBlock)
                    {
                        MemoryMarshal.CreateSpan(ref bufferHead, (colBlock - jj) * rowBlock)
                            .Clear();
                    }
                }
                else
                {
                    ref var colOffset = ref jHead;
                    for (; jj < jBlockSize; jj++)
                    {
                        ref double colPtr = ref Unsafe.Add(ref head, colOffset);

                        int ii = 0;
                        ref int rowOffset = ref iHead;
                        for (; ii < iBlockSize; ii++)
                        {
                            bufferHead = Unsafe.Add(ref colPtr, rowOffset);
                            bufferHead = ref Unsafe.Add(ref bufferHead, 1);
                            rowOffset = ref Unsafe.Add(ref rowOffset, 1);
                        }
                        for (; ii < rowBlock; ii++)
                        {
                            bufferHead = 0.0;
                            bufferHead = ref Unsafe.Add(ref bufferHead, 1);
                        }
                        colOffset = ref Unsafe.Add(ref colOffset, 1);
                    }
                    if (jj < colBlock)
                    {
                        MemoryMarshal.CreateSpan(ref bufferHead, (colBlock - jj) * rowBlock)
                            .Clear();
                    }
                }
            }
        }
    }
}