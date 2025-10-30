using DDLA.BLAS.Managed;
using DDLA.Core;
using DDLA.Factorizations;
using DDLA.Misc;
using DDLA.Misc.Flags;
using DDLA.Misc.Pools;
using DDLA.Utilities;
using static DDLA.BLAS.BlasProvider;

namespace DDLA.Transformations;

public static class Bidiagonaling
{
    public static void Bidiag(MatrixView A, out MatrixView U, out MatrixView V, out VectorView d, out VectorView e)
    {
        var bid = new TwoStageBidiag(A);
        bid.Kernel();
        U = bid.U;
        V = bid.V;
        d = bid.Diag;
        e = bid.SubDiag;
    }

    public static void Bidiag(MatrixView A, MatrixView U, MatrixView V, VectorView d, VectorView e)
    {
        var bid = new TwoStageBidiag(A);
        bid.Kernel();
        bid.U.CopyTo(U);
        bid.V.CopyTo(V);
        bid.Diag.CopyTo(d);
        bid.SubDiag.CopyTo(e);
    }


    public static void Bidiag2(MatrixView A, out MatrixView U, out MatrixView V, out VectorView d, out VectorView e)
    {
        var len = A.Rows;
        var wid = A.Cols;
        if (len < wid)
            throw new ArgumentException("Only support m >= n");
        d = Vector.Create(wid);
        e = Vector.Create(wid - 1);
        U = Matrix.Eyes(len, colMajor: true);
        V = Matrix.Eyes(wid, colMajor: true);
        Bidiag(A, U, V, d, e);
    }

    public static void Bidiag2(MatrixView A, MatrixView U, MatrixView V, VectorView d, VectorView e)
    {
        var len = A.Rows;
        var wid = A.Cols;
        var TUBlockSize = Math.Min(BlockSize, A.Cols);
        var TVBlockSize = Math.Min(BlockSize, A.Cols - 1);
        var TU = Matrix.Create(TUBlockSize, wid, colMajor: true);
        var TV = Matrix.Create(TVBlockSize, wid);
        using var bufferHandle = InternelPool.TakeVector(len, out var buffer); 
        HHStepBlock(A, TU, TV, d, e, buffer);
        BuildUV(A, TU, TV, U, V, TUBlockSize, TVBlockSize);
    }

    // set to 64 or 128 maybe result in crash with aocl-blis
    public static int BlockSize => 128;

    public static void HHStepBlock(MatrixView A, MatrixView TU, MatrixView TV, 
        VectorView d, VectorView e, VectorView buffer)
    {
        var iStart = 0;
        while (iStart < A.Cols)
        {
            var block = Math.Min(BlockSize, A.Cols - iStart);
            var iEnd = iStart + block;
            var AB1 = A[iStart.., iStart..iEnd];
            var A1B = A[iStart..iEnd, iStart..];
            var TUT1 = TU[..block, iStart..iEnd];
            var TVT1 = iEnd == A.Cols ?
                (block == 1 ?
                    TV[..0, iStart..] :
                    TV[..(block - 1), iStart..^1]) :
                TV[..block, iStart..iEnd]; ;

            for (var i = 0; i < block; i++)
            {
                ref var tau = ref TUT1[i, i];
                var t01 = TUT1[..i, i];
                var col = AB1[i.., i];
                var a10 = AB1[i, ..i];
                var a21 = AB1[(i + 1).., i];
                var A20 = AB1[(i + 1).., ..i];
                var right = A[iStart.., (iStart + 1)..];
                Tridiagonaling.BuildHH(col, out d[iStart], out tau);

                var bufferCurrent = buffer[..(A.Cols - iStart - 1)];
                // right -= col * (col^T * right) / tau
                col.LeftMul(1 / tau, right, bufferCurrent);
                right.Rank1(-1, col, bufferCurrent);

                a10.CopyTo(t01);
                a21.LeftMul(A20, 1.0, t01);

                if (iStart < A.Cols - 1 && i < block - 1 || iEnd < A.Cols)
                {
                    tau = ref TVT1[i, i];
                    t01 = TVT1[..i, i];
                    var row = A1B[i, (i + 1)..];
                    var a01 = A1B[..i, i + 1];
                    var a12 = A1B[i, (i + 2)..];
                    var A02 = A1B[..i, (i + 2)..];
                    var bottom = A[(iStart + 1).., (iStart + 1)..];
                    //var VRight = V[.., (i + 1)..]; 
                    Tridiagonaling.BuildHH(row, out e[iStart], out tau);
                    bufferCurrent = buffer[..bottom.Rows];
                    // bottom -= (bottom * row) * row^T / tau
                    bottom.Multify(1 / tau, row, bufferCurrent);
                    bottom.Rank1(-1, bufferCurrent, row);
                    //ApplyUV(row, VRight, tau);

                    a01.CopyTo(t01);
                    A02.Multify(a12, 1.0, t01);
                }
                iStart++;
            }
        }
    }

    public static void BuildUV(MatrixView A, MatrixView TU, MatrixView TV, 
        MatrixView U, MatrixView V, int TUBlockSize, int TVBlockSize)
    {
        var wid = A.Cols;
        var wid2 = wid - 1;
        var Tmp = Matrix.Create(TUBlockSize, wid);

        var T = TU;
        var W = Tmp[..TUBlockSize, ..];
        var Q = U;
        //Crash here
        ApplyQ(A, T, W, Q);

        A = A[..wid2, 1..].T;
        T = TV[.., ..^1];
        W = Tmp[..TVBlockSize, ..wid2];
        Q = V[1.., 1..];
        ApplyQ(A, T, W, Q);
    }

    public static MatrixView GetBiMatrix(MatrixView A, VectorView d, VectorView e)
    {
        var wid = A.Cols;
        MatrixView res = A.EmptyLike();
        int i = 0;
        for (; i < wid - 1; i++)
        {
            res[i, i] = d[i];
            res[i, i + 1] = e[i];
        }
        res[i, i] = d[i];
        return res;
    }

    internal static void ApplyQ(MatrixView A, MatrixView T, MatrixView W, MatrixView Q)
    {
        var aLeft = A.MinDim;
        var aRight = A.MinDim;

        bool left = A.Cols % Math.Min(T.Rows, A.Cols) > 0;
        while (aLeft > 0)
        {
            var block = Math.Min(T.Rows, aLeft);
            if (left)
            {
                block = A.Cols % block;
                left = false;
            }

            aLeft -= block;

            ApplyQStep(A[aLeft.., aLeft..aRight],
                T[..block, aLeft..aRight], W[..block, ..],
                Q[aLeft.., ..], false);

            aRight -= block;
        }
    }

    internal static void ApplyQStep(MatrixView A, MatrixView T, MatrixView W, MatrixView Q, bool trans = false)
    {
        var block = A.Cols;
        var A11 = A[..block, ..];
        var A21 = A[block.., ..];
        var B1 = Q[..block, ..];
        var B2 = Q[block.., ..];

        // B -= A * T^(-1) * A^T * B

        B1.CopyTo(W);

        // X = A^T * B
        // LEFT: B -= A * T^(-1) * X

        // X = A^T * B = A11^T * B1 + A21^T * B2
        TrMM(SideType.Left, UpLo.Lower,
            TransType.OnlyTrans, DiagType.Unit,
            1, A11, W);
        A21.T.Multify(1, B2, 1, W);

        // Y = T^(-1) * X
        // LEFT: B -= A * Y
        TrSM(SideType.Left, UpLo.Upper,
            trans ? TransType.OnlyTrans : TransType.NoTrans,
            DiagType.NonUnit,
            1, T, W);

        // B -= A * Y
        // B2 -= A21 * Y
        DDLA.BLAS.BlasProvider.GeMM(-1, A21, W, 1, B2);
        //A21.Multify(-1, W, 1, B2);

        // B1 -= A11 * Y 
        TrMM(SideType.Left, UpLo.Lower,
            TransType.NoTrans, DiagType.Unit,
            1, A11, W);

        B1.SubtractedBy(W);
    }

}

internal abstract class BidiagBase
{
    protected MatrixView Work { get; }
    public abstract MatrixView U { get; }
    public abstract MatrixView V { get; }
    public VectorView Diag { get; }
    public VectorView SubDiag { get; }
    public abstract void Kernel();
    public virtual Matrix GetBiMatrix()
    {
        var len = Work.Rows;
        var wid = Work.Cols;
        Matrix res = Matrix.Create(len, wid);
        int i = 0;
        for (; i < wid - 1; i++)
        {
            res[i, i] = Diag[i];
            res[i, i + 1] = SubDiag[i];
        }
        res[i, i] = Diag[i];
        return res;
    }

    public BidiagBase(MatrixView orig)
    {
        var len = orig.Rows;
        var wid = orig.Cols;
        if (len < wid)
            throw new ArgumentException("Only support m >= n");
        Work = orig.Clone(colMajor: true);
        Diag = Vector.Create(wid);
        SubDiag = Vector.Create(wid - 1);
    }
}

internal class HHUnbBidiag : BidiagBase
{
    public HHUnbBidiag(MatrixView orig) : base(orig)
    {
        var len = orig.Rows;
        var wid = orig.Cols;
        U = Matrix.Eyes(len).Transpose();
        V = Matrix.Eyes(wid).Transpose();
        Buffer = Vector.Create(Math.Max(len, wid));
    }

    public VectorView Buffer { get; }

    public override MatrixView U { get; }

    public override MatrixView V { get; }

    public override void Kernel()
    {
        for (int i = 0; i < Work.Cols; i++)
        {
            StepU(i);
            if (i < Work.Cols - 1)
                StepV(i);
        }
    }

    public void StepU(int i)
    {
        var col = Work[i.., i];
        var right = Work[i.., (i + 1)..];
        var URight = U[.., i..];
        BidHelper.BuildHH(col, out Diag[i], out var tau1);
        ApplyRight(col, right, tau1);
        ApplyUV(col, URight, tau1);
    }

    public void StepV(int i)
    {
        var row = Work[i, (i + 1)..];
        var bottom = Work[(i + 1).., (i + 1)..];
        var VRight = V[.., (i + 1)..];
        BidHelper.BuildHH(row, out SubDiag[i], out var tau2);
        ApplyBottom(row, bottom, tau2);
        ApplyUV(row, VRight, tau2);
    }

    public void ApplyRight(VectorView col, MatrixView right, double tau)
    {
        var buffer = Buffer[..right.Cols];
        // right -= col * (col^T * right) / tau
        col.LeftMul(1 / tau, right, buffer);
        right.Rank1(-1, col, buffer);
    }

    public void ApplyBottom(VectorView row, MatrixView bottom, double tau)
    {
        var buffer = Buffer[..bottom.Rows];
        // bottom -= (bottom * row) * row^T / tau
        bottom.Multify(1 / tau, row, buffer);
        bottom.Rank1(-1, buffer, row);
    }

    public void ApplyUV(VectorView col, MatrixView UorV, double tau)
    {
        var buffer = Buffer[..UorV.Rows];
        // bottom -= (bottom * col) * col^T / tau
        UorV.Multify(1 / tau, col, buffer);
        UorV.Rank1(-1, buffer, col);
    }
}

internal class TwoStageBidiag : BidiagBase
{
    public const int MaxBlockSize = 32;

    public TwoStageBidiag(MatrixView orig) : base(orig)
    {
        var len = orig.Rows;
        var wid = orig.Cols;
        U = MatrixView.Eyes(len, colMajor: true);
        V = MatrixView.Eyes(wid, colMajor: true);
        TU = MatrixView.Create(TUBlockSize, wid, colMajor: true);
        TV = MatrixView.Create(TUBlockSize, wid - TUBlockSize);
        TU2 = MatrixView.Create(wid, wid);
        TV2 = MatrixView.Create(wid, wid);
    }

    public override MatrixView U { get; }

    public override MatrixView V { get; }

    public MatrixView TU { get; }

    public MatrixView TV { get; }

    public MatrixView TU2 { get; }

    public MatrixView TV2 { get; }

    public int TUBlockSize => Math.Min(MaxBlockSize, Work.Cols);

    public override void Kernel()
    {
        DateTime start = DateTime.Now;
        Ge2Bd();
        var span = DateTime.Now - start;
        Console.WriteLine($"Ge2Bd time out: {span}");

        start = DateTime.Now;
        Bd2Bi();
        span = DateTime.Now - start;
        Console.WriteLine($"Bd2Bi time out: {span}");
        // Extract diagonal and subdiagonal
        Work.Diag.CopyTo(Diag);
        Work[.., 1..].Diag.CopyTo(SubDiag);
    }

    public override Matrix GetBiMatrix() => new(Work);

    #region Ge2Bd
    public void Ge2Bd()
    {
        // Use a blocked algorithm to reduce the matrix to upper band bidiagonal form

        // A buffer to apply qr dec with.
        using var BufferHandle = InternelPool.TakeMatrix(TUBlockSize, U.Cols,
            out var Buffer, init: false);

        var partA = PartitionGrid.Create
            (Work, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        var partTU = PartitionHorizontal.Create
            (TU, 0, SideType.Left,
            out var TU0, out var TU1, out var TU2);

        var partTV = PartitionHorizontal.Create
            (TV, 0, SideType.Left,
            out var TV0, out var TV1, out var TV2);

        while (A22.Cols > 0)
        {
            var block = Math.Min(MaxBlockSize, A22.Cols);

            using var partAStep = partA.Step(block, block);
            using var partTUStep = partTU.Step(block);

            // Reduce the current block column

            // Left QR Dec
            PartUtils.Merge21to11(A11, A21, out var AB1);
            PartUtils.Merge21to11(A12, A22, out var AB2);
            PartUtils.Part11to21(TU1, block, UpLo.Upper,
                out var TUT1, out _);
            PartUtils.Part11to22(Buffer, block, AB2.Cols, Quadrant.TopLeft,
                out var W, out _,
                    out _, out _);
            BidHelper.QRUnblock(AB1, TUT1);
            BidHelper.ApplyQStep(AB1, TUT1, W, AB2, trans: true);

            if (A22.Cols > 0)
            {
                // Right LQ Dec, as transposed QR Dec.
                if (A22.Cols < MaxBlockSize)
                {
                    var block2 = A22.Cols;

                    using var partTVStep = partTV.Step(block2);
                    var AB2T = AB2.T;
                    PartUtils.Part11to12(AB2T, block2, SideType.Left,
                        out var A12T, out var A22T);
                    PartUtils.Part11to22(TV1, block2, block2, Quadrant.TopLeft,
                        out var TVT1, out _,
                               out _, out _);
                    // var TVT1 = TV[..block2, iStart..iEnd2];
                    PartUtils.Part11to22(Buffer, block2, A22T.Cols, Quadrant.TopLeft,
                        out W, out _,
                        out _, out _);
                    BidHelper.QRUnblock(A12T, TVT1);
                    BidHelper.ApplyQStep(A12T, TVT1, W, A22T, trans: true);
                }
                else
                {
                    using var partTVStep = partTV.Step(block);
                    var A12T = A12.T;
                    var A22T = A22.T;
                    PartUtils.Part11to22(TV1, block, block, Quadrant.TopLeft,
                        out var TVT1, out _,
                               out _, out _);
                    PartUtils.Part11to22(Buffer, block, A22.Rows, Quadrant.TopLeft,
                        out W, out _,
                        out _, out _);
                    BidHelper.QRUnblock(A12T, TVT1);
                    BidHelper.ApplyQStep(A12T, TVT1, W, A22T, trans: true);
                }
            }
        }

        // Build U and V for this stage
        var A = Work;
        var Q = U;
        QR.ApplyQlnfc(A, TU, Buffer, Q);
        BlasProvider.Set(DiagType.Unit, UpLo.Lower, 0, Work);

        A = Work[..TV.Cols, TUBlockSize..].T;
        Q = V[.., TUBlockSize..];
        QR.ApplyQlnfc(A, TV, Buffer[.., ..TV.Cols], Q);
        BlasProvider.Set(DiagType.Unit, UpLo.Lower, 0, A);
    }
    #endregion Ge2Bd

    #region Bd2Bi
    public void Bd2Bi()
    {
        for (var i = 0; i < Work.Cols - 1; i++)
        {
            Bd2Bi_StepCycle(Work[i..Work.Cols, i..], TU2[i, i..], TV2[i, i..]);
        }
        for (var i = 0; i < Work.Cols - 1; i++)
        {
            Bd2Bi_BackCycle(Work[i..Work.Cols, i..],
                U[.., i..Work.Cols], V[.., i..], TU2[i, i..], TV2[i, i..]);
        }
    }

    private void Bd2Bi_StepCycle(MatrixView A, VectorView tU, VectorView tV)
    {
        var block = TUBlockSize;
        var iStart = 1;
        var iEnd = iStart + block;
        var iEnd2 = iEnd + block;
        if (A.Rows <= iEnd)
        {
            Bd2Bi_StepSide(A[.., iStart..].T, tV[iStart..]);
            Bd2Bi_StepSide(A[iStart.., iStart..], tU[iStart..]);
        }
        else
        {
            Bd2Bi_StepSide(A[..iEnd, iStart..iEnd].T, tV[iStart..iEnd]);
            while (iEnd2 < A.Rows)
            {
                Bd2Bi_StepSide(A[iStart..iEnd, iStart..iEnd2], tU[iStart..iEnd]);
                Bd2Bi_StepSide(A[iStart..iEnd2, iEnd..iEnd2].T, tV[iEnd..iEnd2]);
                iStart += block;
                iEnd += block;
                iEnd2 += block;
            }
            Bd2Bi_StepSide(A[iStart..iEnd, iStart..], tU[iStart..iEnd]);
            Bd2Bi_StepSide(A[iStart.., iEnd..].T, tV[iEnd..]);
            Bd2Bi_StepSide(A[iEnd.., iEnd..], tU[iEnd..]);
        }
    }

    private void Bd2Bi_BackCycle(MatrixView A,
        MatrixView U, MatrixView V, VectorView tU, VectorView tV)
    {
        var block = TUBlockSize;
        var maxU = (A.Rows - 2) / block + 1;
        var invoker = new Bd2BiBackParallelInvoker(A.Rows, block, U, tU, V, tV);
        var maxThreads = Environment.ProcessorCount / 2;
        var useThreads = Math.Min(maxThreads, maxU);
        ParallelHelperEX.For(0, maxU, invoker, 1, useThreads);
    }

    private static void Bd2Bi_StepSide(MatrixView A, VectorView t)
    {
        VectorView line = A[.., 0];
        MatrixView applied = A[.., 1..];
        ref var chi = ref line[0];
        var xLast = line[1..];
        ref var tau = ref t[0];
        var tLast = t[1..];
        HouseHolder.BuildHouseHolder(ref chi, xLast, out tau);
        HouseHolder.ApplyHouseHolder(SideType.Left, tau, xLast, applied);
        xLast.CopyTo(tLast);
        xLast.Clear();
    }

    private static void Bd2Bi_Back(VectorView t, MatrixView Q)
    {
        var tau = t[0];
        var tLast = t[1..];
        HouseHolder.ApplyHouseHolder(SideType.Left, tau, tLast, Q.T);
    }

    private readonly struct Bd2BiBackParallelInvoker(int maxLen, int blockSize,
        MatrixView U, VectorView tU, MatrixView V, VectorView tV) : IActionEX
    {
        int MaxLen { get; } = maxLen;

        int BlockSize { get; } = blockSize;

        MatrixView U { get; } = U;

        VectorView tU { get; } = tU;

        MatrixView V { get; } = V;
        VectorView tV { get; } = tV;

        public void Invoke(int i)
        {
            var iStart = 1 + i * BlockSize;
            var iEnd = Math.Min(iStart + BlockSize, MaxLen);
            Bd2Bi_Back(tV[iStart..iEnd], V[.., iStart..iEnd]);
            Bd2Bi_Back(tU[iStart..iEnd], U[.., iStart..iEnd]);
        }
    }
    #endregion Bd2Bi
}

internal class BidHelper
{

    #region QR
    public const int MaxBlockSize = 128;

    internal static void QRUnblock(MatrixView A, MatrixView T)
    {
        for (int i = 0; i < A.MinDim; i++)
        {
            ref var tau = ref T[i, i];
            var t01 = T[..i, i];
            ref var alpha = ref A[i, i];
            var a10 = A[i, ..i];
            var a21 = A[(i + 1).., i];
            var A20 = A[(i + 1).., ..i];
            var Ab1 = A[i.., i];
            var Ab2 = A[i.., (i + 1)..];

            BuildHH(Ab1, out tau);
            ApplyHH(Ab1, Ab2, tau);

            a10.CopyTo(t01);
            a21.LeftMul(A20, 1.0, t01);
        }
    }

    internal static void QRBlock(MatrixView A, MatrixView T)
    {
        int iNext = 0;

        for (var i = 0; i < A.MinDim; i = iNext)
        {
            iNext = Math.Min(i + MaxBlockSize, A.MinDim);

            var T1 = T[.., i..iNext];
            var T2 = T[.., iNext..];
            var AB1 = A[i.., i..iNext];
            var AB2 = A[i.., iNext..];

            QRUnblock(AB1, T1);
            ApplyQT(AB1, T1, T2, AB2);
        }
    }

    internal static void ApplyQT(MatrixView A, MatrixView T, MatrixView W, MatrixView Q)
    {
        int block = T.Rows;

        for (int aLeft = 0; aLeft < A.MinDim; aLeft += block)
        {
            block = Math.Min(block, A.MinDim - aLeft);
            var aRight = aLeft + block;

            ApplyQStep(A[aLeft.., aLeft..aRight],
                T[aLeft..aRight, ..block], W[..block, ..Q.Cols],
                Q[aLeft.., ..], trans: true);
        }
    }

    internal static void ApplyQ(MatrixView A, MatrixView T, MatrixView W, MatrixView Q)
    {
        var aLeft = A.MinDim;
        var aRight = A.MinDim;
        var qLeft = A.MinDim;
        var qRight = A.MinDim;

        bool left = A.Cols % Math.Min(T.Rows, A.Cols) > 0;
        var i = 0;
        while (aLeft > 0)
        {
            var block = Math.Min(T.Rows, aLeft);
            if (left)
            {
                block = A.Cols % block;
                left = false;
            }

            aLeft -= block;

            ApplyQStep(A[aLeft.., aLeft..aRight],
                T[..block, aLeft..aRight], W[..block, ..],
                Q[aLeft.., ..], false);

            aRight -= block;
        }
    }

    internal static void ApplyQStep(MatrixView A, MatrixView T, MatrixView W, MatrixView Q, bool trans = false)
    {
        var block = A.Cols;
        var A11 = A[..block, ..];
        var A21 = A[block.., ..];
        var B1 = Q[..block, ..];
        var B2 = Q[block.., ..];

        // B -= A * T^(-1) * A^T * B

        B1.CopyTo(W);

        // X = A^T * B
        // LEFT: B -= A * T^(-1) * X

        // X = A^T * B = A11^T * B1 + A21^T * B2
        TrMM(SideType.Left, UpLo.Lower,
            TransType.OnlyTrans, DiagType.Unit,
            1, A11, W);
        A21.T.Multify(1, B2, 1, W);

        // Y = T^(-1) * X
        // LEFT: B -= A * Y
        TrSM(SideType.Left, UpLo.Upper,
            trans ? TransType.OnlyTrans : TransType.NoTrans,
            DiagType.NonUnit,
            1, T, W);

        // B -= A * Y
        // B2 -= A21 * Y
        GeMM(-1, A21, W, 1, B2);
        //A21.Multify(-1, W, 1, B2);

        // B1 -= A11 * Y 
        TrMM(SideType.Left, UpLo.Lower,
            TransType.NoTrans, DiagType.Unit,
            1, A11, W);

        B1.SubtractedBy(W);
    }
    #endregion QR

    #region HH
    internal static void BuildHH(VectorView A, out double tau)
    {
        ref double a11 = ref A[0];
        var A21 = A[1..];
        double xSq = A21.SumSq();
        double alphaSq = a11 * a11 + xSq;
        double alpha = Math.Sqrt(alphaSq);
        double rho = -Math.Sign(a11) * alpha;
        double miu = a11 - rho;
        A21.Scaled(1 / miu);
        tau = 1 + A21.SumSq();
        a11 = rho;
        tau /= 2;
    }
    public static void BuildHH(VectorView x, out double sigma, out double tau)
    {
        if (x.Length == 0)
            throw new ArgumentException("Vector length must be at least 1.", nameof(x));

        ref var chi = ref x[0];
        var xLast = x[1..];

        double lenLast = xLast.NrmF();

        if (lenLast == 0.0)
        {
            sigma = -chi;
            chi = 1;
            tau = 0.5;
            return;
        }

        double lenx = double.Hypot(chi, lenLast);

        double neg_alpha = double.CopySign(lenx, chi);
        double scale = chi + neg_alpha;

        InvScal(scale, xLast);
        double scaledLenLast = lenLast / Math.Abs(scale);

        chi = 1;
        sigma = -neg_alpha;
        tau = (1 + scaledLenLast * scaledLenLast) / 2;
    }

    internal static void ApplyHH(VectorView A, MatrixView A2, double tau)
    {
        var a21 = A[1..];

        for (var col = 0; col < A2.Cols; col++)
        {
            ref var a12 = ref A2[0, col];
            var a22 = A2[1.., col];

            var w = a12 + a21 * a22;
            w /= tau;
            a12 -= w;
            a22.AddedBy(-w, a21);
        }
    }
    #endregion HH
}