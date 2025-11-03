using DDLA.Misc;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.UFuncs.Operators;
using DDLA.UFuncs;
using DDLA.Misc.Flags;
using System.Diagnostics;
using DDLA.Utilities;
using CommunityToolkit.HighPerformance;
using DDLA.Core;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    public interface IPack: IActionEX
    {

    }

    public static partial class Packing 
    { 
        public readonly struct GEPack: IPack
        {
            public matrix Src { get; }

            public ArraySegment<double> Buffer { get; }

            public int Rows => Src.Rows;

            public int Cols => Src.Cols;

            public int BlockSize { get; }

            public int RowsAligned { get; }

            public GEPack(matrix src, ArraySegment<double> buffer, int block, bool packB = false)
            {
                Src = packB ? src.T : src;
                Buffer = buffer;
                BlockSize = block;

                RowsAligned = Rows.Align(BlockSize);
                Debug.Assert(buffer.Count >= RowsAligned * Cols);
            }

            public void Pack()
            {
                ParallelHelperEX.For(0, RowsAligned / BlockSize, this, 
                    4, Environment.ProcessorCount / 2);
            }

            public void Invoke(int i)
            {
                var length = Cols * BlockSize;
                var offset = i * Cols * BlockSize;
                var dst = Buffer.AsSpan(offset, length);
                var rowStart = i * BlockSize;
                var actualMR = Math.Min(Src.Rows - rowStart, BlockSize);
                var src = Src.SliceSubUncheck(rowStart, actualMR, 0, Cols);
                ref var slot = ref dst[0];
                for (int col = 0; col < Cols; col++)
                {
                    int row = 0;
                    ref var srcRef = ref src.AtUncheck(0, col);
                    for (; row < actualMR; row++)
                    {
                        slot = srcRef;
                        slot = ref Unsafe.Add(ref slot, 1);
                        srcRef = ref Unsafe.Add(ref srcRef, src.RowStride);
                    }
                    for (; row < BlockSize; row++)
                    {
                        slot = 0;
                        slot = ref Unsafe.Add(ref slot, 1);
                    }
                }
            }
        }
    }

    public static void GEMMPack(matrix src, int iStart, int iLength,
        int jStart, int jLength, ArraySegment<double> buffer, int block, bool packB = false)
    {
        var sub = src.SliceSubUncheck(iStart, iLength, jStart, jLength);
        var pack = new Packing.GEPack(sub, buffer, block, packB);
        pack.Pack();
    }

    public static void GEMMPack(matrix src, ArraySegment<double> buffer, int block, bool packB = false)
    {
        var pack = new Packing.GEPack(src, buffer, block, packB);
        pack.Pack();
    }

    public static partial class Packing
    {
        public readonly struct TRPack : IPack
        {
            public bool DiagUnit { get; }

            public UpLo UpLo { get; }

            public int DiagOffset { get; }

            public matrix Src { get; }

            public int MC => Src.Rows;

            public int KC => Src.Cols;

            public ArraySegment<double> Buffer { get; }

            public int BlockSize { get; }

            public int Segments { get; }

            public MArray BufferTensor { get; }

            public TRPack(DiagType unit, UpLo upLo, int diagOffset, in matrix src,
                ArraySegment<double> buffer, int mr, bool packB = false)
            {
                if (packB)
                {
                    UpLo = Transpose(upLo);
                    DiagOffset = -diagOffset;
                    Src = src.T;
                }
                else
                {
                    UpLo = upLo;
                    DiagOffset = diagOffset;
                    Src = src;
                }

                DiagUnit = unit is DiagType.Unit;
                Buffer = buffer;
                BlockSize = mr;
                Segments = (MC + BlockSize - 1) / BlockSize;
                Debug.Assert(buffer.Count >= Segments * BlockSize * KC);
                BufferTensor = new(buffer.Array, buffer.Offset,
                    [Segments, KC, BlockSize]);
            }

            public void Pack()
            {
                ParallelHelperEX.For(0, Segments, this,
                    4, Environment.ProcessorCount / 2);
            }

            public void Invoke(int index)
            {
                var length = KC * BlockSize;
                var dst = Buffer.AsSpan(index * length, length);
                var rowOffset = index * BlockSize;
                var actualMR = Math.Min(Src.Rows - rowOffset, BlockSize);
                var upper = UpLo is UpLo.Upper;
                for (int col = 0; col < KC; col++)
                {
                    int row = 0;
                    for (; row < BlockSize; row++)
                    {
                        BufferTensor[index, col, row] = From(index, row, col);
                    }
                }
            }

            private double From(int index, int row, int col)
            {
                row += index * BlockSize;
                if (row >= MC) return 0.0;
                var offset = col - row - DiagOffset;
                if (offset == 0)
                {
                    if (DiagUnit)
                    {
                        return 1.0;
                    }
                    else
                    {
                        var val = Src[row, col];
                        return val;
                    }
                }
                else if ((UpLo is UpLo.Upper && offset > 0) ||
                   (UpLo is UpLo.Lower && offset < 0))
                {
                    return Src[row, col];
                }
                else
                {
                    return 0.0;
                }
            }
        }
    }

    public static void TRMMPackA(DiagType unit, UpLo upLo, int diagOffset, matrix src,
        int iStart, int iLength, int jStart, int jLength,
        ArraySegment<double> buffer, int block)
    {
        var sub = src.SliceSubUncheck(iStart, iLength, jStart, jLength);
        if (upLo is UpLo.Dense)
        {
            var pack = new Packing.GEPack(sub, buffer, block, false);
            pack.Pack();
        }
        else
        {
            var pack = new Packing.TRPack(unit, upLo, diagOffset, sub,
                buffer, block, false);
            pack.Pack();
        }
    }

    public static void TRMMPackB(DiagType unit, UpLo upLo, int diagOffset, matrix src,
        int iStart, int iLength, int jStart, int jLength,
        ArraySegment<double> buffer, int block)
    {
        var sub = src.SliceSubUncheck(iStart, iLength, jStart, jLength);
        if (upLo is UpLo.Dense)
        {
            var pack = new Packing.GEPack(sub, buffer, block, true);
            pack.Pack();
        }
        else
        {
            var pack = new Packing.TRPack(unit, upLo, diagOffset, sub,
                buffer, block, true);
            pack.Pack();
        }
    }

    public static partial class Packing
    {
        public readonly struct SYPack: IPack
        {
            public UpLo UpLo { get; }

            public int DiagOffset { get; }

            public matrix M { get; }

            public int RowStart { get; }

            public int ColStart { get; }

            public int MC { get; }

            public int KC { get; }

            public ArraySegment<double> Buffer { get; }

            public int MR { get; }

            public int MCAligned { get; }

            public SYPack(UpLo upLo, int diagOffset,
                in matrix src, int rowStart, int rowLength, int colStart, int colLength, 
                ArraySegment<double> buffer, int mr, bool packB = false)
            {
                if (packB)
                {
                    UpLo = Transpose(upLo);
                    DiagOffset = -diagOffset;
                    M = src.T;
                    RowStart = colStart;
                    MC = colLength;
                    ColStart = rowStart;
                    KC = rowLength;
                }
                else
                {
                    UpLo = upLo;
                    DiagOffset = diagOffset;
                    M = src;
                    RowStart = rowStart;
                    MC = rowLength;
                    ColStart = colStart;
                    KC = colLength;
                }

                Buffer = buffer;
                MR = mr;
                MCAligned = (MC + MR - 1) / MR * MR;
                Debug.Assert(buffer.Count >= MCAligned * KC);
            }

            public void Pack()
            {
                ParallelHelperEX.For(0, MCAligned / MR, this,
                    4, Environment.ProcessorCount / 2);
            }

            public void Invoke(int i)
            {
                var length = KC * MR;
                var dst = Buffer.AsSpan(i * length, length);
                var rowOffset = i * MR;
                var actualMR = Math.Min(M.Rows - rowOffset, MR);
                var upper = UpLo is UpLo.Upper;
                ref var slot = ref dst[0];
                for (int col = 0; col < KC; col++)
                {
                    var jFull = ColStart + col;
                    var iFull = RowStart + rowOffset;
                    var offset = jFull - iFull - DiagOffset;
                    int row = 0;
                    ref var srcRef = ref M.AtUncheck(iFull, jFull);
                    if (upper)
                    {
                        if (offset < 0)
                        {
                            srcRef = ref M.AtUncheck(jFull, iFull);
                        }

                        for (; row < actualMR; row++)
                        {
                            slot = srcRef;
                            var stride = offset > 0 ? M.RowStride : M.ColStride;
                            srcRef = ref Unsafe.Add(ref srcRef, stride);
                            slot = ref Unsafe.Add(ref slot, 1);
                            offset--;
                        }
                    }
                    else
                    {
                        if (offset > 0)
                        {
                            srcRef = ref M.AtUncheck(jFull, iFull);
                        }

                        for (; row < actualMR; row++)
                        {
                            slot = srcRef;
                            var stride = offset <= 0 ? M.RowStride : M.ColStride;
                            srcRef = ref Unsafe.Add(ref srcRef, stride);
                            slot = ref Unsafe.Add(ref slot, 1);
                            offset--;
                        }
                    }
                    for (; row < MR; row++)
                    {
                        slot = 0;
                        slot = ref Unsafe.Add(ref slot, 1);
                    }
                }
            }

        }
    }

    public static void SYMMPack(UpLo upLo, matrix src,
        int iStart, int iLength, int jStart, int jLength, 
        ArraySegment<double> buffer, int block, bool packB = false)
    {
        if (upLo is UpLo.Dense)
            GEMMPack(src, iStart, iLength,
                jStart, jLength, buffer, block, packB);
        else
        {
            var pack = new Packing.SYPack(upLo, 0, src,
                iStart, iLength, jStart, jLength,
                buffer, block, packB);
            pack.Pack();
        }
    }

}
