using DDLA.BLAS.Managed;
using DDLA.Misc.Flags;
using SimpleExample.LAFFExercise.QRs;
using DDLA.Factorizations;
using DDLA.Transformations;
using DDLA.Utilities;
using DDLA.Misc;
using DDLA.Misc.Pools;

namespace SimpleExample.SVD.Bidiag;

public class TwoStageBidiag : BidiagBase
{
    public const int MaxBlockSize = 32;

    public TwoStageBidiag(Matrix orig) : base(orig)
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
            HHUTQR.QRUnblock(AB1, TUT1);
            HHUTQR.ApplyQStep(AB1, TUT1, W, AB2, trans: true);

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
                    HHUTQR.QRUnblock(A12T, TVT1);
                    HHUTQR.ApplyQStep(A12T, TVT1, W, A22T, trans: true);
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
                    HHUTQR.QRUnblock(A12T, TVT1);
                    HHUTQR.ApplyQStep(A12T, TVT1, W, A22T, trans: true);
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
            while(iEnd2 < A.Rows)
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
