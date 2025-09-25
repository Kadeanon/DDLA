using DDLA.Factorizations;
using DDLA.Misc;
using DDLA.Misc.Flags;
using DDLA.Transformations;

using static DDLA.BLAS.BlasProvider;

namespace SimpleExample.SymmEVD.Tradiag;

public class HHUTTradiag: TridiagBase
{
    private MatrixView T { get; }

    public override MatrixView Q => Work;

    internal static int BlockSize => 16;

    public HHUTTradiag(Matrix orig) : base(orig)
    {
        var len = orig.Rows;

        T = Matrix.Create(Math.Min(BlockSize, len), Work.Rows);
    }

    public override void Kernel()
    {
        var partA = PartitionGrid.Create
            (Work, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);
        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Left,
            out var T0, out var T1, out var T2);
        int index = 0;
        while (A22.Rows > 0)
        {
            int block = Math.Min(T.Rows, A22.Rows);
            var indexNext = index + block;
            var ABR = A22;
            using var partAStep = partA.Step(block, block);
            using var partTStep = partT.Step(block);

            var subDiagEnd = Math.Min(indexNext, SubDiag.Length);
            var d = Diag[index..indexNext];
            var e = SubDiag[index..subDiagEnd];
            Step(ABR, T1[..block, ..], d, e);
            index = indexNext;
        }
        FormQ(Work, T);
    }

    internal static void Step(MatrixView A, MatrixView T, 
        VectorView d, VectorView e)
    {
        int ARows = A.Rows;
        int TRows = T.Rows;

        VectorView tmp = Vector.Create(ARows);

        for (var i = 0; i < TRows; i++)
        {
            d[i] = A[i, i];
            if (i < ARows - 1)
            {
                var A20 = A[(i + 1).., ..i];
                var a21 = A[(i + 1).., i];
                var A22 = A[(i + 1).., (i + 1)..];
                var t01 = T[..i, i];
                ref double tau11 = ref T[i, i];
                var p = tmp[(i + 1)..];

                HHUnbTradiag.BuildHH(a21, out var sigma, out tau11);

                SyMV(UpLo.Lower,
                    1, A22, a21,
                    0, p);
                var beta = a21 * p / (2 * tau11);
                p.AddedBy(-beta, a21);
                p.InvScaled(tau11);
                A22.Rank2(UpLo.Lower,
                    -1.0, a21, p);

                a21.LeftMul(A20, t01);
                a21.GetHeadRef() = sigma;
                e[i] = sigma;
            }
        }
    }

    internal static void FormQ(MatrixView A, MatrixView T)
    {
        for (int j = A.Rows - 2; j > 0; --j)
        {
            A[(j + 1).., j - 1].CopyTo(A[(j + 1).., j]);
        }
        A[0, ..].Fill(0);
        A[.., 0].Fill(0);
        A.Diag.Fill(1);

        A = A[1.., 1..];
        T = T[.., ..^1];
        A.MakeTr(UpLo.Lower);

        var slice = ..A.Cols;
        if (T.Cols > A.Cols)
            T = T[.., slice];

        var W = Matrix.Create(T.Rows, A.Cols);

        int block, b_alg;
        int m_BR, n_BR;

        b_alg = T.Rows;
        m_BR = A.Rows - A.Cols;
        n_BR = 0;

        var partA = PartitionGrid.Create
            (A, m_BR, n_BR, Quadrant.BottomRight,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);

        var partT = PartitionHorizontal.Create
            (T, 0, SideType.Right,
            out var T0, out var T1, out var T2);

        while (T0.Cols > 0)
        {
            block = Math.Min(b_alg, A00.MinDim);

            if (T2.Cols == 0 && T.Cols % b_alg > 0)
                block = T.Cols % b_alg;

            var ABR = A22;
            using var partAStep = partA.Step(block, block);
            using var partTStep = partT.Step(block);

            var TT1 = T1[..block, ..];
            var WTL = W[..block, ..A12.Cols];

            if (ABR.Rows != 0)
            {
                PartUtils.Merge21to11
                    (A11,
                     A21, out var AB1);
                PartUtils.Merge21to11
                    (A12,
                     A22, out var AB2);

                QR.ApplyQlnfc(AB1, TT1, WTL, AB2);
                FormQUnblock(AB1, TT1);
            }
            else
                FormQUnblock(A11, TT1);
        }
    }

    internal static void FormQUnblock(MatrixView A, MatrixView T)
    {
        int min_m_n = A.MinDim;
        int i;

        for (i = min_m_n - 1; i >= 0; --i)
        {
            var slice = (i + 1)..;
            ref var alphA11 = ref A[i, i];
            var a21 = A[slice, i];
            var a12t = A[i, slice];
            var A22 = A[slice, slice];

            ref double tau11 = ref T[i, i];

            HouseHolder.ApplyHouseHolder(SideType.Left,
                ref tau11, a21, a12t, A22);

            alphA11 = 1 - 1 / tau11;

            InvScal(-tau11, a21);
        }
    }

}
