using DDLA.Core;
using DDLA.Misc.Flags;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace DDLA.Misc;

public static class PartUtils
{
    #region Horizontal Partitioning
    public static void Part11to12(MatrixView a, int np, SideType side, out MatrixView aL, out MatrixView aR)
    {
        if (a.Cols < np)
            throw new ArgumentException($"Matrix a has not enough columns to partition into 1x2 blocks with {np} columns in the second block.");
        if (side == SideType.Left)
            Part11to12Uncheck_Left(a, np, out aL, out aR);
        else
            Part11to12Uncheck_Right(a, np, out aL, out aR);
    }

    public static void Part12to13(ref MatrixView aL, out MatrixView aM, ref MatrixView aR, int np, SideType side)
    {
        if (side == SideType.Left)
        {
            if (aL.Cols < np)
                throw new ArgumentException($"Matrix aL has not enough columns to partition into 1x3 blocks with {np} columns in the middle block.");
            Part11to12Uncheck_Right(aL, np, out aL, out aM);
        }
        else
        {
            if (aR.Cols < np)
                throw new ArgumentException($"Matrix aR has not enough columns to partition into 1x3 blocks with {np} columns in the middle block.");
            Part11to12Uncheck_Left(aR, np, out aM, out aR);
        }
    }

    public static void Comb13to12(ref MatrixView aL, in MatrixView aM, ref MatrixView aR, SideType side)
    {
        if (side == SideType.Left)
        {
            if (!CheckCombHorizontal(aL, aM))
                throw new ArgumentException($"Matrix aL does not match the expected structure for combining with aM.");
            Comb12to11Uncheck_Left(ref aL, in aM);
        }
        else
        {
            if (!CheckCombHorizontal(aM, aR))
                throw new ArgumentException($"Matrix aR does not match the expected structure for combining with aM.");
            Comb12to11Uncheck_Right(in aM, ref aR);
        }
    }

    public static void Merge12to11(in MatrixView aL, in MatrixView aR, out MatrixView a)
    {
        if (!CheckCombHorizontal(aL, aR))
            throw new ArgumentException($"Matrix aL does not match the expected structure for combining with aR.");

        a = aL;
        Comb12to11Uncheck_Left(ref a, in aR);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to12Uncheck_Left(MatrixView a, int np, out MatrixView aL, out MatrixView aR)
    {
        int anoCols = a.Cols - np;
        aL = a.SliceSubUncheck(0, a.Rows, 0, np);
        aR = a.SliceSubUncheck(0, a.Rows, np, anoCols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to12Uncheck_Right(MatrixView a, int np, out MatrixView aL, out MatrixView aR)
    {
        int anoCols = a.Cols - np;
        aL = a.SliceSubUncheck(0, a.Rows, 0, anoCols);
        aR = a.SliceSubUncheck(0, a.Rows, anoCols, np);
    }

    internal static void Comb12to11Uncheck_Left(ref MatrixView a0, in MatrixView a1)
        => a0 = new(a0.Data, a0.Offset, a0.Rows, a0.Cols + a1.Cols, a0.RowStride, a0.ColStride);

    internal static void Comb12to11Uncheck_Right(in MatrixView a0, ref MatrixView a1)
        => a1 = new(a0.Data, a0.Offset, a0.Rows, a0.Cols + a1.Cols, a0.RowStride, a0.ColStride);
    #endregion Horizontal Partitioning

    #region Vertical Partitioning
    public static void Part11to21(MatrixView a, int mp, UpLo uplo, out MatrixView aT, out MatrixView aB)
    {
        if (a.Rows < mp)
            throw new ArgumentException($"Matrix a has not enough columns to partition into 1x2 blocks with {mp} columns in the second block.");
        if (uplo == UpLo.Upper)
            Part11to21Uncheck_Top(a, mp, out aT, out aB);
        else
            Part11to21Uncheck_Bottom(a, mp, out aT, out aB);
    }

    public static void Part21to31(ref MatrixView aT, out MatrixView aM, ref MatrixView aB, int Mp, UpLo uplo)
    {
        if (uplo == UpLo.Upper)
        {
            if (aT.Rows < Mp)
                throw new ArgumentException($"Matrix aT has not enough rows to partition into 3x1 blocks with {Mp} rows in the middle block.");
            Part11to21Uncheck_Bottom(aT, Mp, out aT, out aM);
        }
        else
        {
            if (aB.Rows < Mp)
                throw new ArgumentException($"Matrix aB has not enough rows to partition into 3x1 blocks with {Mp} rows in the middle block.");
            Part11to21Uncheck_Top(aB, Mp, out aM, out aB);
        }
    }

    public static void Comb31to21(ref MatrixView aT, in MatrixView aM, ref MatrixView aB, UpLo uplo)
    {
        if (uplo == UpLo.Upper)
        {
            if (!CheckCombVertical(aT, aM))
                throw new ArgumentException($"Matrix aT does not match the expected structure for combining with aM.");
            Comb21to11Uncheck_Top(ref aT, in aM);
        }
        else
        {
            if (!CheckCombVertical(aM, aB))
                throw new ArgumentException($"Matrix aB does not match the expected structure for combining with aM.");
            Comb21to11Uncheck_Bottom(in aM, ref aB);
        }
    }

    public static void Merge21to11(in MatrixView aT, in MatrixView aB, out MatrixView a)
    {
        if (!CheckCombVertical(aT, aB))
            throw new ArgumentException($"Matrix aB does not match the expected structure for combining with aB.");
        a = aT;
        Comb21to11Uncheck_Top(ref a, in aB);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to21Uncheck_Top(MatrixView a, int mp, out MatrixView aT, out MatrixView aB)
    {
        int anoRows = a.Rows - mp;
        aT = a.SliceSubUncheck(0, mp, 0, a.Cols);
        aB = a.SliceSubUncheck(mp, anoRows, 0, a.Cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to21Uncheck_Bottom(MatrixView a, int mp, out MatrixView aT, out MatrixView aB)
    {
        int anoRows = a.Rows - mp;
        aT = a.SliceSubUncheck(0, anoRows, 0, a.Cols);
        aB = a.SliceSubUncheck(anoRows, mp, 0, a.Cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb21to11Uncheck_Top(ref MatrixView a0, in MatrixView a1)
        => a0 = new(a0.Data, a0.Offset, a0.Rows + a1.Rows, a0.Cols, a0.RowStride, a0.ColStride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb21to11Uncheck_Bottom(in MatrixView a0, ref MatrixView a1)
        => a1 = new(a0.Data, a0.Offset, a0.Rows + a1.Rows, a0.Cols, a0.RowStride, a0.ColStride);
    #endregion Vertical Partitioning

    #region 2D Partitioning
    public static void Part11to22(
        MatrixView a, int mp, int np, Quadrant quad,
        out MatrixView a00, out MatrixView a01,
        out MatrixView a10, out MatrixView a11)
    {
        if (a.Rows < mp || a.Cols < np)
            throw new ArgumentException($"Matrix a has not enough rows or columns to partition into 2x2 blocks with {mp} rows and {np} columns.");
        switch (quad)
        {
            case Quadrant.TopLeft:
                Part11to22Uncheck_TopLeft(a, mp, np, out a00, out a01, out a10, out a11);
                break;
            case Quadrant.TopRight:
                Part11to22Uncheck_TopRight(a, mp, np, out a00, out a01, out a10, out a11);
                break;
            case Quadrant.BottomLeft:
                Part11to22Uncheck_BottomLeft(a, mp, np, out a00, out a01, out a10, out a11);
                break;
            default:
                //case Quadrant.LowerRight:
                Part11to22Uncheck_BottomRight(a, mp, np, out a00, out a01, out a10, out a11);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Part22to33Uncheck(
        ref MatrixView a00, out MatrixView a01, ref MatrixView a02,
        out MatrixView a10, out MatrixView a11, out MatrixView a12,
        ref MatrixView a20, out MatrixView a21, ref MatrixView a22,
        int Mp, int Np, Quadrant quad)
    {
        switch (quad)
        {
            case Quadrant.TopLeft:
                Part11to22Uncheck_BottomRight(a00, Mp, Np, out a00, out a01,
                                                           out a10, out a11);
                Part11to12Uncheck_Right(a20, Np, out a20, out a21);
                Part11to21Uncheck_Bottom(a02, Mp, out a02, out a12);
                break;
            case Quadrant.TopRight:
                Part11to22Uncheck_BottomLeft(a02, Mp, Np, out a01, out a02,
                                                          out a11, out a12);
                Part11to12Uncheck_Left(a22, Np, out a21, out a22);
                Part11to21Uncheck_Bottom(a00, Mp, out a00, out a10);
                break;
            case Quadrant.BottomLeft:
                Part11to22Uncheck_TopRight(a20, Mp, Np, out a10, out a11,
                                                        out a20, out a21);
                Part11to12Uncheck_Right(a00, Np, out a00, out a01);
                Part11to21Uncheck_Top(a22, Mp, out a12, out a22);
                break;
            default:
                //case Quadrant.BottomRight:
                Part11to22Uncheck_TopLeft(a22, Mp, Np, out a11, out a12,
                                                       out a21, out a22);
                Part11to12Uncheck_Left(a02, Np, out a01, out a02);
                Part11to21Uncheck_Top(a20, Mp, out a10, out a20);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Comb33to22Uncheck(
        ref MatrixView a00, in MatrixView a01, ref MatrixView a02,
         in MatrixView a10, in MatrixView a11, in MatrixView a12,
        ref MatrixView a20, in MatrixView a21, ref MatrixView a22,
        Quadrant quad)
    {
        switch (quad)
        {
            case Quadrant.TopLeft:
                Comb33to22Uncheck_TopLeft(ref a00, in a01, ref a02,
                                                  in a10, in a11, in a12,
                                                 ref a20, in a21, ref a22);
                break;
            case Quadrant.TopRight:
                Comb33to22Uncheck_TopRight(ref a00, in a01, ref a02,
                                                   in a10, in a11, in a12,
                                                  ref a20, in a21, ref a22);
                break;
            case Quadrant.BottomLeft:
                Comb33to22Uncheck_BottomLeft(ref a00, in a01, ref a02,
                                                  in a10, in a11, in a12,
                                                 ref a20, in a21, ref a22);
                break;
            default:
                //case Quadrant.LowerRight:
                Comb33to22Uncheck_BottomRight(ref a00, in a01, ref a02,
                                                   in a10, in a11, in a12,
                                                  ref a20, in a21, ref a22);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Comb22to11Uncheck(
        ref MatrixView a00, ref MatrixView a01,
        ref MatrixView a10, ref MatrixView a11,
        Quadrant quad)
    {
        switch (quad)
        {
            case Quadrant.TopLeft:
                Comb22to11Uncheck_TopLeft(
                    ref a00, in a01,
                     in a10, in a11);
                break;
            case Quadrant.TopRight:
                Comb22to11Uncheck_TopRight(
                    in a00, ref a01,
                    in a10, in a11);
                break;
            case Quadrant.BottomLeft:
                Comb22to11Uncheck_BottomLeft(
                     in a00, in a01,
                    ref a10, in a11);
                break;
            default:
                //case Quadrant.LowerRight:
                Comb22to11Uncheck_BottomRight(
                    in a00, in a01,
                    in a10, ref a11);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_TopLeft(
        MatrixView a, int mp, int np,
        out MatrixView a00, out MatrixView a01,
        out MatrixView a10, out MatrixView a11)
    {
        int anoRows = a.Rows - mp;
        int anoCols = a.Cols - np;
        a00 = a.SliceSubUncheck(0, mp, 0, np);
        a01 = a.SliceSubUncheck(0, mp, np, anoCols);
        a10 = a.SliceSubUncheck(mp, anoRows, 0, np);
        a11 = a.SliceSubUncheck(mp, anoRows, np, anoCols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_TopRight(
        MatrixView a, int mp, int np,
        out MatrixView a00, out MatrixView a01,
        out MatrixView a10, out MatrixView a11)
    {
        int anoRows = a.Rows - mp;
        int anoCols = a.Cols - np;
        a00 = a.SliceSubUncheck(0, mp, 0, anoCols);
        a01 = a.SliceSubUncheck(0, mp, anoCols, np);
        a10 = a.SliceSubUncheck(mp, anoRows, 0, anoCols);
        a11 = a.SliceSubUncheck(mp, anoRows, anoCols, np);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_BottomLeft(
        MatrixView a, int mp, int np,
        out MatrixView a00, out MatrixView a01,
        out MatrixView a10, out MatrixView a11)
    {
        int anoRows = a.Rows - mp;
        int anoCols = a.Cols - np;
        a00 = a.SliceSubUncheck(0, anoRows, 0, np);
        a01 = a.SliceSubUncheck(0, anoRows, np, anoCols);
        a10 = a.SliceSubUncheck(anoRows, mp, 0, np);
        a11 = a.SliceSubUncheck(anoRows, mp, np, anoCols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_BottomRight(
        MatrixView a, int mp, int np,
        out MatrixView a00, out MatrixView a01,
        out MatrixView a10, out MatrixView a11)
    {
        int anoRows = a.Rows - mp;
        int anoCols = a.Cols - np;
        a00 = a.SliceSubUncheck(0, anoRows, 0, anoCols);
        a01 = a.SliceSubUncheck(0, anoRows, anoCols, np);
        a10 = a.SliceSubUncheck(anoRows, mp, 0, anoCols);
        a11 = a.SliceSubUncheck(anoRows, mp, anoCols, np);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_TopLeft(
        ref MatrixView a00, in MatrixView a01, ref MatrixView a02,
         in MatrixView a10, in MatrixView a11, in MatrixView a12,
        ref MatrixView a20, in MatrixView a21, ref MatrixView a22)
    {
        Comb22to11Uncheck_TopLeft(ref a00, in a01,
                                   in a10, in a11);
        Comb12to11Uncheck_Left(ref a20, in a21);
        Comb21to11Uncheck_Top(ref a02, in a12);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_TopRight(
        ref MatrixView a00, in MatrixView a01, ref MatrixView a02,
         in MatrixView a10, in MatrixView a11, in MatrixView a12,
        ref MatrixView a20, in MatrixView a21, ref MatrixView a22)
    {
        Comb22to11Uncheck_TopRight(in a01, ref a02,
                                         in a11, in a12);
        Comb12to11Uncheck_Right(in a21, ref a22);
        Comb21to11Uncheck_Top(ref a00, in a10);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_BottomRight(
        ref MatrixView a00, in MatrixView a01, ref MatrixView a02,
         in MatrixView a10, in MatrixView a11, in MatrixView a12,
        ref MatrixView a20, in MatrixView a21, ref MatrixView a22)
    {
        Comb12to11Uncheck_Right(in a01, ref a02);
        Comb22to11Uncheck_BottomRight(in a11, in a12,
                                         in a21, ref a22);
        Comb21to11Uncheck_Bottom(in a10, ref a20);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_BottomLeft(
        ref MatrixView a00, in MatrixView a01, ref MatrixView a02,
         in MatrixView a10, in MatrixView a11, in MatrixView a12,
        ref MatrixView a20, in MatrixView a21, ref MatrixView a22)
    {
        Comb12to11Uncheck_Left(ref a00, in a01);
        Comb22to11Uncheck_BottomLeft(in a10, in a11,
                                         ref a20, in a21);
        Comb21to11Uncheck_Bottom(in a12, ref a22);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_TopLeft(
        ref MatrixView a00, in MatrixView a01,
         in MatrixView a10, in MatrixView a11)
    {
        a00 = new(a00.Data, a00.Offset, a00.Rows + a10.Rows, a00.Cols + a01.Cols, a00.RowStride, a00.ColStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_TopRight(
        in MatrixView a00, ref MatrixView a01,
        in MatrixView a10, in MatrixView a11)
    {
        a01 = new(a00.Data, a00.Offset, a00.Rows + a10.Rows, a00.Cols + a01.Cols, a00.RowStride, a00.ColStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_BottomLeft(
         in MatrixView a00, in MatrixView a01,
        ref MatrixView a10, in MatrixView a11)
    {
        a10 = new(a00.Data, a00.Offset, a00.Rows + a10.Rows, a00.Cols + a01.Cols, a00.RowStride, a00.ColStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_BottomRight(
       in MatrixView a00, in MatrixView a01,
       in MatrixView a10, ref MatrixView a11)
    {
        a11 = new(a00.Data, a00.Offset, a00.Rows + a11.Rows, a00.Cols + a01.Cols, a00.RowStride, a00.ColStride);
    }

    #endregion

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool CheckCombHorizontal(in MatrixView aL, in MatrixView aR)
        => aL.Data == aR.Data &&
           aL.Rows == aR.Rows &&
           aL.Offset + aL.ColStride * aL.Cols == aR.Offset;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool CheckCombVertical(in MatrixView aT, in MatrixView aB)
        => aT.Data == aB.Data &&
           aT.Cols == aB.Cols &&
           aT.Offset + aT.RowStride * aT.Rows == aB.Offset;
}

public readonly ref struct PartitionHorizontal : IDisposable
{
    readonly ref MatrixView a00;
    readonly ref MatrixView a01;
    readonly ref MatrixView a02;
    readonly SideType side;

    private PartitionHorizontal(
        ref MatrixView a00,
        ref MatrixView a01,
        ref MatrixView a02,
        SideType side)
    {
        this.a00 = ref a00;
        this.a01 = ref a01;
        this.a02 = ref a02;
        this.side = side;
    }

    public static PartitionHorizontal Create(
        in MatrixView a, int initBlock, SideType side,
        [UnscopedRef] out MatrixView a0,
        [UnscopedRef] out MatrixView a1,
        [UnscopedRef] out MatrixView a2,
        bool backward = false)
    {
        PartUtils.Part11to12
            (a, initBlock, side, out a0, out a2);
        a1 = default;
        if (!backward)
            side = Reverse(side);
        return new PartitionHorizontal(
            ref a0, ref a1, ref a2, side);
    }

    public PartitionHorizontal Step(int block)
    {
        PartUtils.
            Part12to13(ref a00, out a01, ref a02,
            block, side);
        return this;
    }

    public PartitionHorizontal Step()
        => Step(1);

    public void Dispose()
    {
        PartUtils.Comb13to12(
            ref a00, in a01, ref a02,
            Reverse(side));
    }

    private static SideType Reverse(SideType side)
        => side switch
        {
            SideType.Left => SideType.Right,
            _ => SideType.Left
        };
}

public readonly ref struct PartitionVertical : IDisposable
{
    readonly ref MatrixView a00;
    readonly ref MatrixView a10;
    readonly ref MatrixView a20;
    readonly UpLo uplo;

    public PartitionVertical(
        ref MatrixView a00,
        ref MatrixView a10,
        ref MatrixView a20,
        UpLo uplo)
    {
        this.a00 = ref a00;
        this.a10 = ref a10;
        this.a20 = ref a20;
        this.uplo = uplo;
    }

    public static PartitionVertical Create(
        in MatrixView a, int initBlock, UpLo uplo,
        [UnscopedRef] out MatrixView a0,
        [UnscopedRef] out MatrixView a1,
        [UnscopedRef] out MatrixView a2,
        bool backward = false)
    {
        PartUtils.Part11to21(a, initBlock, uplo,
            out a0, out a2);
        a1 = default;
        if (!backward)
            uplo = Reverse(uplo);
        return new PartitionVertical(
            ref a0, ref a1, ref a2, uplo);
    }

    public PartitionVertical Step(int block)
    {
        PartUtils.Part21to31(ref a00, out a10, ref a20,
            block, uplo);
        return this;
    }

    public PartitionVertical Step()
        => Step(1);

    public void Dispose()
    {
        PartUtils.Comb31to21(
            ref a00, in a10, ref a20,
            Reverse(uplo));
    }

    private static UpLo Reverse(UpLo uplo)
        => uplo switch
        {
            UpLo.Upper => UpLo.Lower,
            _ => UpLo.Upper
        };
}

public readonly ref struct PartitionGrid : IDisposable
{
    readonly ref MatrixView a00;
    readonly ref MatrixView a01;
    readonly ref MatrixView a02;
    readonly ref MatrixView a10;
    readonly ref MatrixView a11;
    readonly ref MatrixView a12;
    readonly ref MatrixView a20;
    readonly ref MatrixView a21;
    readonly ref MatrixView a22;
    readonly Quadrant quad;
    public PartitionGrid(
        ref MatrixView a00,
        ref MatrixView a01,
        ref MatrixView a02,
        ref MatrixView a10,
        ref MatrixView a11,
        ref MatrixView a12,
        ref MatrixView a20,
        ref MatrixView a21,
        ref MatrixView a22,
        Quadrant quad)
    {
        this.a00 = ref a00;
        this.a01 = ref a01;
        this.a02 = ref a02;
        this.a10 = ref a10;
        this.a11 = ref a11;
        this.a12 = ref a12;
        this.a20 = ref a20;
        this.a21 = ref a21;
        this.a22 = ref a22;
        this.quad = quad;
    }
    public static PartitionGrid Create(
        in MatrixView a, int initBlockRow,
        int initBlockCol, Quadrant quad,
        [UnscopedRef] out MatrixView a00,
        [UnscopedRef] out MatrixView a01,
        [UnscopedRef] out MatrixView a02,
        [UnscopedRef] out MatrixView a10,
        [UnscopedRef] out MatrixView a11,
        [UnscopedRef] out MatrixView a12,
        [UnscopedRef] out MatrixView a20,
        [UnscopedRef] out MatrixView a21,
        [UnscopedRef] out MatrixView a22,
        bool backward = false)
    {
        PartUtils.Part11to22(
            a, initBlockRow, initBlockCol, quad,
            out a00, out a02,
            out a20, out a22);
        a01 = default;
        a10 = default;
        a11 = default;
        a12 = default;
        a21 = default;

        if (!backward)
            quad = Reverse(quad);

        return new PartitionGrid(
            ref a00, ref a01, ref a02,
            ref a10, ref a11, ref a12,
            ref a20, ref a21, ref a22, quad);
    }

    public PartitionGrid Step(
        int blockRow, int blockCol)
    {
        PartUtils.Part22to33Uncheck(
            ref a00, out a01, ref a02,
            out a10, out a11, out a12,
            ref a20, out a21, ref a22,
            blockRow, blockCol, quad);
        return this;
    }

    public PartitionGrid Step()
        => Step(1, 1);

    public void Dispose()
    {
        PartUtils.Comb33to22Uncheck(
            ref a00, in a01, ref a02,
             in a10, in a11, in a12,
            ref a20, in a21, ref a22,
            Reverse(quad));
    }

    private static Quadrant Reverse(Quadrant quad)
    {
        return quad switch
        {
            Quadrant.TopLeft => Quadrant.BottomRight,
            Quadrant.TopRight => Quadrant.BottomLeft,
            Quadrant.BottomLeft => Quadrant.TopRight,
            _ => Quadrant.TopLeft, // BottomRight
        };
    }
}
