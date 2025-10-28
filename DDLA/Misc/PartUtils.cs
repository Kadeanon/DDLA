using DDLA.Core;
using DDLA.Misc.Flags;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace DDLA.Misc;

public static class PartUtils
{
    #region Horizontal Partitioning
    public static void Part11to12(MatrixView A, int np, SideType side, out MatrixView AL, out MatrixView AR)
    {
        if (A.Cols < np)
            throw new ArgumentException($"Matrix A has not enough columns to partition into 1x2 blocks with {np} columns in the second block.");
        if (side == SideType.Left)
            Part11to12Uncheck_Left(A, np, out AL, out AR);
        else
            Part11to12Uncheck_Right(A, np, out AL, out AR);
    }

    public static void Part12to13(ref MatrixView AL, out MatrixView AM, ref MatrixView AR, int np, SideType side)
    {
        if (side == SideType.Left)
        {
            if (AL.Cols < np)
                throw new ArgumentException($"Matrix AL has not enough columns to partition into 1x3 blocks with {np} columns in the middle block.");
            Part11to12Uncheck_Right(AL, np, out AL, out AM);
        }
        else
        {
            if (AR.Cols < np)
                throw new ArgumentException($"Matrix AR has not enough columns to partition into 1x3 blocks with {np} columns in the middle block.");
            Part11to12Uncheck_Left(AR, np, out AM, out AR);
        }
    }

    public static void Comb13to12(ref MatrixView AL, in MatrixView AM, ref MatrixView AR, SideType side)
    {
        if (side == SideType.Left)
        {
            if (!CheckCombHorizontal(AL, AM))
                throw new ArgumentException($"Matrix AL does not match the expected structure for combining with AM.");
            Comb12to11Uncheck_Left(ref AL, in AM);
        }
        else
        {
            if (!CheckCombHorizontal(AM, AR))
                throw new ArgumentException($"Matrix AR does not match the expected structure for combining with AM.");
            Comb12to11Uncheck_Right(in AM, ref AR);
        }
    }

    public static void Merge12to11(in MatrixView AL, in MatrixView AR, out MatrixView A)
    {
        if (!CheckCombHorizontal(AL, AR))
            throw new ArgumentException($"Matrix AL does not match the expected structure for combining with AR.");

        A = AL;
        Comb12to11Uncheck_Left(ref A, in AR);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to12Uncheck_Left(MatrixView A, int np, out MatrixView AL, out MatrixView AR)
    {
        int anoCols = A.Cols - np;
        AL = A.SliceSubUncheck(0, A.Rows, 0, np);
        AR = A.SliceSubUncheck(0, A.Rows, np, anoCols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to12Uncheck_Right(MatrixView A, int np, out MatrixView AL, out MatrixView AR)
    {
        int anoCols = A.Cols - np;
        AL = A.SliceSubUncheck(0, A.Rows, 0, anoCols);
        AR = A.SliceSubUncheck(0, A.Rows, anoCols, np);
    }

    internal static void Comb12to11Uncheck_Left(ref MatrixView A0, in MatrixView A1)
        => A0 = new(A0.Data, A0.Offset, A0.Rows, A0.Cols + A1.Cols, A0.RowStride, A0.ColStride);

    internal static void Comb12to11Uncheck_Right(in MatrixView A0, ref MatrixView A1)
        => A1 = new(A0.Data, A0.Offset, A0.Rows, A0.Cols + A1.Cols, A0.RowStride, A0.ColStride);
    #endregion Horizontal Partitioning

    #region Vertical Partitioning
    public static void Part11to21(MatrixView A, int mp, UpLo uplo, out MatrixView AT, out MatrixView AB)
    {
        if (A.Rows < mp)
            throw new ArgumentException($"Matrix A has not enough columns to partition into 1x2 blocks with {mp} columns in the second block.");
        if (uplo == UpLo.Upper)
            Part11to21Uncheck_Top(A, mp, out AT, out AB);
        else
            Part11to21Uncheck_Bottom(A, mp, out AT, out AB);
    }

    public static void Part21to31(ref MatrixView AT, out MatrixView AM, ref MatrixView AB, int Mp, UpLo uplo)
    {
        if (uplo == UpLo.Upper)
        {
            if (AT.Rows < Mp)
                throw new ArgumentException($"Matrix AT has not enough rows to partition into 3x1 blocks with {Mp} rows in the middle block.");
            Part11to21Uncheck_Bottom(AT, Mp, out AT, out AM);
        }
        else
        {
            if (AB.Rows < Mp)
                throw new ArgumentException($"Matrix AB has not enough rows to partition into 3x1 blocks with {Mp} rows in the middle block.");
            Part11to21Uncheck_Top(AB, Mp, out AM, out AB);
        }
    }

    public static void Comb31to21(ref MatrixView AT, in MatrixView AM, ref MatrixView AB, UpLo uplo)
    {
        if (uplo == UpLo.Upper)
        {
            if (!CheckCombVertical(AT, AM))
                throw new ArgumentException($"Matrix AT does not match the expected structure for combining with AM.");
            Comb21to11Uncheck_Top(ref AT, in AM);
        }
        else
        {
            if (!CheckCombVertical(AM, AB))
                throw new ArgumentException($"Matrix AB does not match the expected structure for combining with AM.");
            Comb21to11Uncheck_Bottom(in AM, ref AB);
        }
    }

    public static void Merge21to11(in MatrixView AT, in MatrixView AB, out MatrixView A)
    {
        if (!CheckCombVertical(AT, AB))
            throw new ArgumentException($"Matrix AB does not match the expected structure for combining with AB.");
        A = AT;
        Comb21to11Uncheck_Top(ref A, in AB);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to21Uncheck_Top(MatrixView A, int mp, out MatrixView AT, out MatrixView AB)
    {
        int anoRows = A.Rows - mp;
        AT = A.SliceSubUncheck(0, mp, 0, A.Cols);
        AB = A.SliceSubUncheck(mp, anoRows, 0, A.Cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to21Uncheck_Bottom(MatrixView A, int mp, out MatrixView AT, out MatrixView AB)
    {
        int anoRows = A.Rows - mp;
        AT = A.SliceSubUncheck(0, anoRows, 0, A.Cols);
        AB = A.SliceSubUncheck(anoRows, mp, 0, A.Cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb21to11Uncheck_Top(ref MatrixView A0, in MatrixView A1)
        => A0 = new(A0.Data, A0.Offset, A0.Rows + A1.Rows, A0.Cols, A0.RowStride, A0.ColStride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb21to11Uncheck_Bottom(in MatrixView A0, ref MatrixView A1)
        => A1 = new(A0.Data, A0.Offset, A0.Rows + A1.Rows, A0.Cols, A0.RowStride, A0.ColStride);
    #endregion Vertical Partitioning

    #region 2D Partitioning
    public static void Part11to22(
        MatrixView A, int mp, int np, Quadrant quad,
        out MatrixView A00, out MatrixView A01,
        out MatrixView A10, out MatrixView A11)
    {
        if (A.Rows < mp || A.Cols < np)
            throw new ArgumentException($"Matrix A has not enough rows or columns to partition into 2x2 blocks with {mp} rows and {np} columns.");
        switch (quad)
        {
            case Quadrant.TopLeft:
                Part11to22Uncheck_TopLeft(A, mp, np, out A00, out A01, out A10, out A11);
                break;
            case Quadrant.TopRight:
                Part11to22Uncheck_TopRight(A, mp, np, out A00, out A01, out A10, out A11);
                break;
            case Quadrant.BottomLeft:
                Part11to22Uncheck_BottomLeft(A, mp, np, out A00, out A01, out A10, out A11);
                break;
            default:
                //case Quadrant.LowerRight:
                Part11to22Uncheck_BottomRight(A, mp, np, out A00, out A01, out A10, out A11);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Part22to33Uncheck(
        ref MatrixView A00, out MatrixView A01, ref MatrixView A02,
        out MatrixView A10, out MatrixView A11, out MatrixView A12,
        ref MatrixView A20, out MatrixView A21, ref MatrixView A22,
        int Mp, int Np, Quadrant quad)
    {
        switch (quad)
        {
            case Quadrant.TopLeft:
                Part11to22Uncheck_BottomRight(A00, Mp, Np, out A00, out A01,
                                                           out A10, out A11);
                Part11to12Uncheck_Right(A20, Np, out A20, out A21);
                Part11to21Uncheck_Bottom(A02, Mp, out A02, out A12);
                break;
            case Quadrant.TopRight:
                Part11to22Uncheck_BottomLeft(A02, Mp, Np, out A01, out A02,
                                                          out A11, out A12);
                Part11to12Uncheck_Left(A22, Np, out A21, out A22);
                Part11to21Uncheck_Bottom(A00, Mp, out A00, out A10);
                break;
            case Quadrant.BottomLeft:
                Part11to22Uncheck_TopRight(A20, Mp, Np, out A10, out A11,
                                                        out A20, out A21);
                Part11to12Uncheck_Right(A00, Np, out A00, out A01);
                Part11to21Uncheck_Top(A22, Mp, out A12, out A22);
                break;
            default:
                //case Quadrant.BottomRight:
                Part11to22Uncheck_TopLeft(A22, Mp, Np, out A11, out A12,
                                                       out A21, out A22);
                Part11to12Uncheck_Left(A02, Np, out A01, out A02);
                Part11to21Uncheck_Top(A20, Mp, out A10, out A20);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Comb33to22Uncheck(
        ref MatrixView A00, in MatrixView A01, ref MatrixView A02,
         in MatrixView A10, in MatrixView A11, in MatrixView A12,
        ref MatrixView A20, in MatrixView A21, ref MatrixView A22,
        Quadrant quad)
    {
        switch (quad)
        {
            case Quadrant.TopLeft:
                Comb33to22Uncheck_TopLeft(ref A00, in A01, ref A02,
                                                  in A10, in A11, in A12,
                                                 ref A20, in A21, ref A22);
                break;
            case Quadrant.TopRight:
                Comb33to22Uncheck_TopRight(ref A00, in A01, ref A02,
                                                   in A10, in A11, in A12,
                                                  ref A20, in A21, ref A22);
                break;
            case Quadrant.BottomLeft:
                Comb33to22Uncheck_BottomLeft(ref A00, in A01, ref A02,
                                                  in A10, in A11, in A12,
                                                 ref A20, in A21, ref A22);
                break;
            default:
                //case Quadrant.LowerRight:
                Comb33to22Uncheck_BottomRight(ref A00, in A01, ref A02,
                                                   in A10, in A11, in A12,
                                                  ref A20, in A21, ref A22);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Comb22to11Uncheck(
        ref MatrixView A00, ref MatrixView A01,
        ref MatrixView A10, ref MatrixView A11,
        Quadrant quad)
    {
        switch (quad)
        {
            case Quadrant.TopLeft:
                Comb22to11Uncheck_TopLeft(
                    ref A00, in A01,
                     in A10, in A11);
                break;
            case Quadrant.TopRight:
                Comb22to11Uncheck_TopRight(
                    in A00, ref A01,
                    in A10, in A11);
                break;
            case Quadrant.BottomLeft:
                Comb22to11Uncheck_BottomLeft(
                     in A00, in A01,
                    ref A10, in A11);
                break;
            default:
                //case Quadrant.LowerRight:
                Comb22to11Uncheck_BottomRight(
                    in A00, in A01,
                    in A10, ref A11);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_TopLeft(
        MatrixView A, int mp, int np,
        out MatrixView A00, out MatrixView A01,
        out MatrixView A10, out MatrixView A11)
    {
        int anoRows = A.Rows - mp;
        int anoCols = A.Cols - np;
        A00 = A.SliceSubUncheck(0, mp, 0, np);
        A01 = A.SliceSubUncheck(0, mp, np, anoCols);
        A10 = A.SliceSubUncheck(mp, anoRows, 0, np);
        A11 = A.SliceSubUncheck(mp, anoRows, np, anoCols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_TopRight(
        MatrixView A, int mp, int np,
        out MatrixView A00, out MatrixView A01,
        out MatrixView A10, out MatrixView A11)
    {
        int anoRows = A.Rows - mp;
        int anoCols = A.Cols - np;
        A00 = A.SliceSubUncheck(0, mp, 0, anoCols);
        A01 = A.SliceSubUncheck(0, mp, anoCols, np);
        A10 = A.SliceSubUncheck(mp, anoRows, 0, anoCols);
        A11 = A.SliceSubUncheck(mp, anoRows, anoCols, np);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_BottomLeft(
        MatrixView A, int mp, int np,
        out MatrixView A00, out MatrixView A01,
        out MatrixView A10, out MatrixView A11)
    {
        int anoRows = A.Rows - mp;
        int anoCols = A.Cols - np;
        A00 = A.SliceSubUncheck(0, anoRows, 0, np);
        A01 = A.SliceSubUncheck(0, anoRows, np, anoCols);
        A10 = A.SliceSubUncheck(anoRows, mp, 0, np);
        A11 = A.SliceSubUncheck(anoRows, mp, np, anoCols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Part11to22Uncheck_BottomRight(
        MatrixView A, int mp, int np,
        out MatrixView A00, out MatrixView A01,
        out MatrixView A10, out MatrixView A11)
    {
        int anoRows = A.Rows - mp;
        int anoCols = A.Cols - np;
        A00 = A.SliceSubUncheck(0, anoRows, 0, anoCols);
        A01 = A.SliceSubUncheck(0, anoRows, anoCols, np);
        A10 = A.SliceSubUncheck(anoRows, mp, 0, anoCols);
        A11 = A.SliceSubUncheck(anoRows, mp, anoCols, np);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_TopLeft(
        ref MatrixView A00, in MatrixView A01, ref MatrixView A02,
         in MatrixView A10, in MatrixView A11, in MatrixView A12,
        ref MatrixView A20, in MatrixView A21, ref MatrixView A22)
    {
        Comb22to11Uncheck_TopLeft(ref A00, in A01,
                                   in A10, in A11);
        Comb12to11Uncheck_Left(ref A20, in A21);
        Comb21to11Uncheck_Top(ref A02, in A12);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_TopRight(
        ref MatrixView A00, in MatrixView A01, ref MatrixView A02,
         in MatrixView A10, in MatrixView A11, in MatrixView A12,
        ref MatrixView A20, in MatrixView A21, ref MatrixView A22)
    {
        Comb22to11Uncheck_TopRight(in A01, ref A02,
                                         in A11, in A12);
        Comb12to11Uncheck_Right(in A21, ref A22);
        Comb21to11Uncheck_Top(ref A00, in A10);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_BottomRight(
        ref MatrixView A00, in MatrixView A01, ref MatrixView A02,
         in MatrixView A10, in MatrixView A11, in MatrixView A12,
        ref MatrixView A20, in MatrixView A21, ref MatrixView A22)
    {
        Comb12to11Uncheck_Right(in A01, ref A02);
        Comb22to11Uncheck_BottomRight(in A11, in A12,
                                         in A21, ref A22);
        Comb21to11Uncheck_Bottom(in A10, ref A20);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb33to22Uncheck_BottomLeft(
        ref MatrixView A00, in MatrixView A01, ref MatrixView A02,
         in MatrixView A10, in MatrixView A11, in MatrixView A12,
        ref MatrixView A20, in MatrixView A21, ref MatrixView A22)
    {
        Comb12to11Uncheck_Left(ref A00, in A01);
        Comb22to11Uncheck_BottomLeft(in A10, in A11,
                                         ref A20, in A21);
        Comb21to11Uncheck_Bottom(in A12, ref A22);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_TopLeft(
        ref MatrixView A00, in MatrixView A01,
         in MatrixView A10, in MatrixView A11)
    {
        A00 = new(A00.Data, A00.Offset, A00.Rows + A10.Rows, A00.Cols + A01.Cols, A00.RowStride, A00.ColStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_TopRight(
        in MatrixView A00, ref MatrixView A01,
        in MatrixView A10, in MatrixView A11)
    {
        A01 = new(A00.Data, A00.Offset, A00.Rows + A10.Rows, A00.Cols + A01.Cols, A00.RowStride, A00.ColStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_BottomLeft(
         in MatrixView A00, in MatrixView A01,
        ref MatrixView A10, in MatrixView A11)
    {
        A10 = new(A00.Data, A00.Offset, A00.Rows + A10.Rows, A00.Cols + A01.Cols, A00.RowStride, A00.ColStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Comb22to11Uncheck_BottomRight(
       in MatrixView A00, in MatrixView A01,
       in MatrixView A10, ref MatrixView A11)
    {
        A11 = new(A00.Data, A00.Offset, A00.Rows + A11.Rows, A00.Cols + A01.Cols, A00.RowStride, A00.ColStride);
    }

    #endregion

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool CheckCombHorizontal(in MatrixView AL, in MatrixView AR)
        => AL.Data == AR.Data &&
           AL.Rows == AR.Rows &&
           AL.Offset + AL.ColStride * AL.Cols == AR.Offset;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool CheckCombVertical(in MatrixView AT, in MatrixView AB)
        => AT.Data == AB.Data &&
           AT.Cols == AB.Cols &&
           AT.Offset + AT.RowStride * AT.Rows == AB.Offset;
}

public readonly ref struct PartitionHorizontal : IDisposable
{
    readonly ref MatrixView A00;
    readonly ref MatrixView A01;
    readonly ref MatrixView A02;
    readonly SideType side;

    private PartitionHorizontal(
        ref MatrixView A00,
        ref MatrixView A01,
        ref MatrixView A02,
        SideType side)
    {
        this.A00 = ref A00;
        this.A01 = ref A01;
        this.A02 = ref A02;
        this.side = side;
    }

    public static PartitionHorizontal Create(
        in MatrixView A, int initBlock, SideType side,
        [UnscopedRef] out MatrixView A0,
        [UnscopedRef] out MatrixView A1,
        [UnscopedRef] out MatrixView A2,
        bool backward = false)
    {
        PartUtils.Part11to12
            (A, initBlock, side, out A0, out A2);
        A1 = default;
        if (!backward)
            side = Reverse(side);
        return new PartitionHorizontal(
            ref A0, ref A1, ref A2, side);
    }

    public static PartitionHorizontal FromLeft(
        in MatrixView A,
        [UnscopedRef] out MatrixView A0,
        [UnscopedRef] out MatrixView A1,
        [UnscopedRef] out MatrixView A2)
    {
        PartUtils.Part11to12
            (A, 0, SideType.Left, out A0, out A2);
        A1 = default;
        return new PartitionHorizontal(
            ref A0, ref A1, ref A2, SideType.Right);
    }

    public static PartitionHorizontal FromRight(
        in MatrixView A,
        [UnscopedRef] out MatrixView A0,
        [UnscopedRef] out MatrixView A1,
        [UnscopedRef] out MatrixView A2)
    {
        PartUtils.Part11to12
            (A, A.Cols, SideType.Left, out A0, out A2);
        A1 = default;
        return new PartitionHorizontal(
            ref A0, ref A1, ref A2, SideType.Left);
    }

    public PartitionHorizontal Step(int block)
    {
        PartUtils.
            Part12to13(ref A00, out A01, ref A02,
            block, side);
        return this;
    }

    public PartitionHorizontal Step()
        => Step(1);

    public void Dispose()
    {
        PartUtils.Comb13to12(
            ref A00, in A01, ref A02,
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
    readonly ref MatrixView A00;
    readonly ref MatrixView A10;
    readonly ref MatrixView A20;
    readonly UpLo uplo;

    public PartitionVertical(
        ref MatrixView A00,
        ref MatrixView A10,
        ref MatrixView A20,
        UpLo uplo)
    {
        this.A00 = ref A00;
        this.A10 = ref A10;
        this.A20 = ref A20;
        this.uplo = uplo;
    }

    public static PartitionVertical Create(
        in MatrixView A, int initBlock, UpLo uplo,
        [UnscopedRef] out MatrixView A0,
        [UnscopedRef] out MatrixView A1,
        [UnscopedRef] out MatrixView A2,
        bool backward = false)
    {
        PartUtils.Part11to21(A, initBlock, uplo,
            out A0, out A2);
        A1 = default;
        if (!backward)
            uplo = Reverse(uplo);
        return new PartitionVertical(
            ref A0, ref A1, ref A2, uplo);
    }

    public static PartitionVertical FromTop(
        in MatrixView A,
        [UnscopedRef] out MatrixView A0,
        [UnscopedRef] out MatrixView A1,
        [UnscopedRef] out MatrixView A2)
    {
        PartUtils.Part11to21(A, 0, UpLo.Upper,
            out A0, out A2);
        A1 = default;
        return new PartitionVertical(
            ref A0, ref A1, ref A2, UpLo.Lower);
    }

    public static PartitionVertical FromBottom(
        in MatrixView A,
        [UnscopedRef] out MatrixView A0,
        [UnscopedRef] out MatrixView A1,
        [UnscopedRef] out MatrixView A2)
    {
        PartUtils.Part11to21(A, A.Rows, UpLo.Upper,
            out A0, out A2);
        A1 = default;
        return new PartitionVertical(
            ref A0, ref A1, ref A2, UpLo.Upper);
    }

    public PartitionVertical Step(int block)
    {
        PartUtils.Part21to31(ref A00, out A10, ref A20,
            block, uplo);
        return this;
    }

    public PartitionVertical Step()
        => Step(1);

    public void Dispose()
    {
        PartUtils.Comb31to21(
            ref A00, in A10, ref A20,
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
    readonly ref MatrixView A00;
    readonly ref MatrixView A01;
    readonly ref MatrixView A02;
    readonly ref MatrixView A10;
    readonly ref MatrixView A11;
    readonly ref MatrixView A12;
    readonly ref MatrixView A20;
    readonly ref MatrixView A21;
    readonly ref MatrixView A22;
    readonly Quadrant quad;
    public PartitionGrid(
        ref MatrixView A00,
        ref MatrixView A01,
        ref MatrixView A02,
        ref MatrixView A10,
        ref MatrixView A11,
        ref MatrixView A12,
        ref MatrixView A20,
        ref MatrixView A21,
        ref MatrixView A22,
        Quadrant quad)
    {
        this.A00 = ref A00;
        this.A01 = ref A01;
        this.A02 = ref A02;
        this.A10 = ref A10;
        this.A11 = ref A11;
        this.A12 = ref A12;
        this.A20 = ref A20;
        this.A21 = ref A21;
        this.A22 = ref A22;
        this.quad = quad;
    }
    public static PartitionGrid Create(
        in MatrixView A, int initBlockRow,
        int initBlockCol, Quadrant quad,
        [UnscopedRef] out MatrixView A00,
        [UnscopedRef] out MatrixView A01,
        [UnscopedRef] out MatrixView A02,
        [UnscopedRef] out MatrixView A10,
        [UnscopedRef] out MatrixView A11,
        [UnscopedRef] out MatrixView A12,
        [UnscopedRef] out MatrixView A20,
        [UnscopedRef] out MatrixView A21,
        [UnscopedRef] out MatrixView A22,
        bool backward = false)
    {
        PartUtils.Part11to22(
            A, initBlockRow, initBlockCol, quad,
            out A00, out A02,
            out A20, out A22);
        A01 = default;
        A10 = default;
        A11 = default;
        A12 = default;
        A21 = default;

        if (!backward)
            quad = Reverse(quad);

        return new PartitionGrid(
            ref A00, ref A01, ref A02,
            ref A10, ref A11, ref A12,
            ref A20, ref A21, ref A22, quad);
    }

    public static PartitionGrid FromTopLeft(
        in MatrixView A, 
        [UnscopedRef] out MatrixView A00,
        [UnscopedRef] out MatrixView A01,
        [UnscopedRef] out MatrixView A02,
        [UnscopedRef] out MatrixView A10,
        [UnscopedRef] out MatrixView A11,
        [UnscopedRef] out MatrixView A12,
        [UnscopedRef] out MatrixView A20,
        [UnscopedRef] out MatrixView A21,
        [UnscopedRef] out MatrixView A22)
    {
        PartUtils.Part11to22(
            A, 0, 0, Quadrant.TopLeft,
            out A00, out A02,
            out A20, out A22);
        A01 = default;
        A10 = default;
        A11 = default;
        A12 = default;
        A21 = default;

        return new PartitionGrid(
            ref A00, ref A01, ref A02,
            ref A10, ref A11, ref A12,
            ref A20, ref A21, ref A22, Quadrant.BottomRight);
    }

    public static PartitionGrid FromBottomRight(
        in MatrixView A,
        [UnscopedRef] out MatrixView A00,
        [UnscopedRef] out MatrixView A01,
        [UnscopedRef] out MatrixView A02,
        [UnscopedRef] out MatrixView A10,
        [UnscopedRef] out MatrixView A11,
        [UnscopedRef] out MatrixView A12,
        [UnscopedRef] out MatrixView A20,
        [UnscopedRef] out MatrixView A21,
        [UnscopedRef] out MatrixView A22)
    {
        PartUtils.Part11to22(
            A, A.Rows, A.Cols, Quadrant.TopLeft,
            out A00, out A02,
            out A20, out A22);
        A01 = default;
        A10 = default;
        A11 = default;
        A12 = default;
        A21 = default;

        return new PartitionGrid(
            ref A00, ref A01, ref A02,
            ref A10, ref A11, ref A12,
            ref A20, ref A21, ref A22, Quadrant.TopLeft);
    }

    public PartitionGrid Step(
        int blockRow, int blockCol)
    {
        PartUtils.Part22to33Uncheck(
            ref A00, out A01, ref A02,
            out A10, out A11, out A12,
            ref A20, out A21, ref A22,
            blockRow, blockCol, quad);
        return this;
    }

    public PartitionGrid Step()
        => Step(1, 1);

    public void Dispose()
    {
        PartUtils.Comb33to22Uncheck(
            ref A00, in A01, ref A02,
             in A10, in A11, in A12,
            ref A20, in A21, ref A22,
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
