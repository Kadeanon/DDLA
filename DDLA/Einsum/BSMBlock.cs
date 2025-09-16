using DDLA.Core;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace DDLA.Einsum;

internal readonly partial struct BSMBlock
{
    internal readonly MArray data;
    internal readonly int offset;
    internal readonly Memory<int> rowScatters;
    public readonly int rowLength;
    public readonly int rowBlock;
    internal readonly Memory<int> colScatters;
    public readonly int colLength;
    public readonly int colBlock;

    public BSMBlock(MArray data, int offset, Memory<int> rowScatters, int rowLength, int rowBlock,
        Memory<int> colScatters, int colLength, int colBlock)
    {
        this.data = data;
        this.offset = offset;
        this.rowScatters = rowScatters;
        this.rowLength = rowLength;
        this.rowBlock = rowBlock;
        this.colScatters = colScatters;
        this.colLength = colLength;
        this.colBlock = colBlock;
    }

    public unsafe readonly void Prefetch(int ii, int ir, int jj, int jr, bool perferCol)
    {
        if (!Sse.IsSupported)
        {
            return;
        }
        ref double originHead = ref GetHeadRef();
        var rowSpan = rowScatters.Span;
        var colSpan = colScatters.Span;
        ref int rowToken = ref rowSpan[ii / rowBlock * (rowBlock + 1)];
        ref int rowHead = ref Unsafe.Add(ref rowToken, 1);
        ref int colToken = ref colSpan[jj / colBlock * (colBlock + 1)];
        ref int colHead = ref Unsafe.Add(ref colToken, 1);
        bool commonRow = rowToken != 0;
        bool commonCol = colToken != 0;

        if (perferCol)
        {
            for (int jjr = 0; jjr < jr; jjr++)
            {
                var colOffset = colToken == 0 ?
                    Unsafe.Add(ref colHead, jjr) :
                    colHead + colToken * jjr;
                ref var ptr = ref Unsafe.Add(ref originHead, colOffset);
                Sse.Prefetch1(Unsafe.AsPointer(
                    ref Unsafe.Add(ref ptr, rowHead)));
            }
        }
        else
        {
            if (commonRow)
            {
                ref var ptr = ref Unsafe.Add(ref originHead, rowHead + colHead);
                for (int iir = 0; iir < ir; iir++)
                {
                    Sse.Prefetch1(Unsafe.AsPointer(
                        ref ptr));
                    ptr = ref Unsafe.Add(ref ptr, rowToken);
                }
            }
            else
            {
                ref var ptr = ref Unsafe.Add(ref originHead, colHead);
                for (int iir = 0; iir < ir; iir++)
                {
                    Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref originHead,
                        Unsafe.Add(ref rowHead, iir))));
                }
            }
        }
    }

    public readonly void UnpackScale2(double alpha, int ii, int ir, int jj, int jr, Span<double> buffer, bool perferCol)
    {
        Debug.Assert(buffer.Length >= rowBlock * colBlock);
        ref double bufferHead = ref buffer[0];
        ref double originHead = ref GetHeadRef();
        var rowSpan = rowScatters.Span;
        var colSpan = colScatters.Span;
        ref int rowToken = ref rowSpan[ii / rowBlock * (rowBlock + 1)];
        ref int rowHead = ref Unsafe.Add(ref rowToken, 1);
        ref int colToken = ref colSpan[jj / colBlock * (colBlock + 1)];
        ref int colHead = ref Unsafe.Add(ref colToken, 1);
        bool commonRow = rowToken != 0;
        bool commonCol = colToken != 0;

        if (perferCol)
        {
            for (int jjr = 0; jjr < jr; jjr++)
            {
                var colOffset = colToken == 0 ?
                    Unsafe.Add(ref colHead, jjr) :
                    colHead + colToken * jjr;
                ref var ptr = ref Unsafe.Add(ref originHead, colOffset);
                ref var bufferPtr = ref bufferHead;
                for (int iir = 0; iir < ir; iir++)
                {
                    var rowOffset = rowToken == 0 ?
                        Unsafe.Add(ref rowHead, iir) :
                        rowHead + rowToken * iir;
                    ref var srcPtr = ref Unsafe.Add(ref ptr, rowOffset);
                    srcPtr = alpha * bufferPtr;
                    bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                }
                bufferHead = ref Unsafe.Add(ref bufferHead, rowBlock);
            }
        }
        else
        {
            if (commonRow)
            {
                ref var ptr = ref Unsafe.Add(ref originHead, rowHead);
                if (commonCol)
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var bufferPtr = ref bufferHead;
                        ref var origPtr = ref Unsafe.Add(ref ptr, colHead);
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            origPtr = bufferPtr * alpha;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                            origPtr = ref Unsafe.Add(ref origPtr, colToken);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                        ptr = ref Unsafe.Add(ref ptr, rowToken);
                    }
                }
                else
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var bufferPtr = ref bufferHead;
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            Unsafe.Add(ref ptr, Unsafe.Add(ref colHead, jjr))
                                = alpha * bufferPtr;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                        ptr = ref Unsafe.Add(ref ptr, rowToken);
                    }
                }
            }
            else
            {
                if (commonCol)
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var ptr = ref Unsafe.Add(ref originHead,
                            Unsafe.Add(ref rowHead, iir) + colHead);
                        ref var bufferPtr = ref bufferHead;
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            ptr = alpha * bufferPtr;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                            ptr = ref Unsafe.Add(ref ptr, colToken);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                    }
                }
                else
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var ptr = ref Unsafe.Add(ref originHead,
                            Unsafe.Add(ref rowHead, iir));
                        ref var bufferPtr = ref bufferHead;
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            Unsafe.Add(ref ptr,
                                Unsafe.Add(ref colHead, jjr)) =
                                alpha * bufferPtr;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                    }
                }
            }
        }
    }

    public readonly void UnpackAxpy(double alpha, int ii, int ir, int jj, int jr, Span<double> buffer, bool perferCol)
    {
        Debug.Assert(buffer.Length >= rowBlock * colBlock);
        ref double bufferHead = ref buffer[0];
        ref double originHead = ref GetHeadRef();
        var rowSpan = rowScatters.Span;
        var colSpan = colScatters.Span;
        ref int rowToken = ref rowSpan[ii / rowBlock * (rowBlock + 1)];
        ref int rowHead = ref Unsafe.Add(ref rowToken, 1);
        ref int colToken = ref colSpan[jj / colBlock * (colBlock + 1)];
        ref int colHead = ref Unsafe.Add(ref colToken, 1);
        bool commonRow = rowToken != 0;
        bool commonCol = colToken != 0;

        if (perferCol)
        {
            for (int jjr = 0; jjr < jr; jjr++)
            {
                var colOffset = colToken == 0 ?
                    Unsafe.Add(ref colHead, jjr) :
                    colHead + colToken * jjr;
                ref var ptr = ref Unsafe.Add(ref originHead, colOffset);
                ref var bufferPtr = ref bufferHead;
                for (int iir = 0; iir < ir; iir++)
                {
                    var rowOffset = rowToken == 0 ?
                        Unsafe.Add(ref rowHead, iir) :
                        rowHead + rowToken * iir;
                    ref var srcPtr = ref Unsafe.Add(ref ptr, rowOffset);
                    srcPtr += alpha * bufferPtr;
                    bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                }
                bufferHead = ref Unsafe.Add(ref bufferHead, rowBlock);
            }
        }
        else
        {
            if (commonRow)
            {
                ref var ptr = ref Unsafe.Add(ref originHead, rowHead);
                if (commonCol)
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var bufferPtr = ref bufferHead;
                        ref var origPtr = ref Unsafe.Add(ref ptr, colHead);
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            origPtr += bufferPtr * alpha;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                            origPtr = ref Unsafe.Add(ref origPtr, colToken);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                        ptr = ref Unsafe.Add(ref ptr, rowToken);
                    }
                }
                else
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var bufferPtr = ref bufferHead;
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            Unsafe.Add(ref ptr, Unsafe.Add(ref colHead, jjr))
                                += alpha * bufferPtr;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                        ptr = ref Unsafe.Add(ref ptr, rowToken);
                    }
                }
            }
            else
            {
                if (commonCol)
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var ptr = ref Unsafe.Add(ref originHead,
                            Unsafe.Add(ref rowHead, iir) + colHead);
                        ref var bufferPtr = ref bufferHead;
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            ptr += alpha * bufferPtr;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                            ptr = ref Unsafe.Add(ref ptr, colToken);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                    }
                }
                else
                {
                    for (int iir = 0; iir < ir; iir++)
                    {
                        ref var ptr = ref Unsafe.Add(ref originHead,
                            Unsafe.Add(ref rowHead, iir));
                        ref var bufferPtr = ref bufferHead;
                        for (int jjr = 0; jjr < jr; jjr++)
                        {
                            Unsafe.Add(ref ptr,
                                Unsafe.Add(ref colHead, jjr)) +=
                                alpha * bufferPtr;
                            bufferPtr = ref Unsafe.Add(ref bufferPtr, 1);
                        }
                        bufferHead = ref Unsafe.Add(ref bufferHead, colBlock);
                    }
                }
            }
        }
    }

    internal readonly void Pack(Memory<double> buffer)
    {
        PackParallel pack = new(this, buffer);
        pack.Pack();
    }

    internal readonly int GetRowOffset(int index)
    {
        int head = index / rowBlock * (rowBlock + 1);
        int left = index % rowBlock;
        ref int token = ref rowScatters.Span[(int)head];
        ref int first = ref Unsafe.Add(ref token, 1);
        int val;
        if (token == 0)
        {
            val = Unsafe.Add(ref first, left);
        }
        else
        {
            val = first + left * token;
        }
        return val;
    }

    internal int GetColOffset(int index)
    {
        int head = index / colBlock * (colBlock + 1);
        int left = index % colBlock;
        ref int token = ref colScatters.Span[(int)head];
        ref int first = ref Unsafe.Add(ref token, 1);
        int val;
        if (token == 0)
        {
            val = Unsafe.Add(ref first, left);
        }
        else
        {
            val = first + left * token;
        }
        return val;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ref double GetHeadRef()
    {
        Span<int> zeros = stackalloc int[data.Rank];
        return ref Unsafe.Add(ref data[zeros], offset);
    }

    public readonly ref double Local(int row, int col)
    {
        int rowOffset = GetRowOffset(row);
        int colOffset = GetColOffset(col);
        ref double head = ref GetHeadRef();
        return ref Unsafe.Add(ref head, rowOffset + colOffset);
    }
}