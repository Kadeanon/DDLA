using DDLA.Core;
using DDLA.Factorizations;
using DDLA.Misc;
using DDLA.Misc.Flags;
using DDLA.Misc.Pools;
using static DDLA.BLAS.Managed.BlasProvider;

namespace DDLA.Transformations;

public static class Bidiagonaling
{
    public static void Bidiag(MatrixView A, out MatrixView U, out MatrixView V, out VectorView d, out VectorView e)
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

    public static void Bidiag(MatrixView A, MatrixView U, MatrixView V, VectorView d, VectorView e)
    {
        var len = A.Rows;
        var wid = A.Cols;
        var TUBlockSize = Math.Min(BlockSize, A.Cols);
        var TVBlockSize = Math.Min(BlockSize, A.Cols - 1);
        var TU = Matrix.Create(TUBlockSize, wid, colMajor: true);
        var TV = Matrix.Create(TVBlockSize, wid);
        var buffer = Vector.Create(len);
        HHStepBlock(A, TU, TV, d, e, buffer);
        BuildUV(A, TU, TV, U, V, TUBlockSize, TVBlockSize);
    }

    // set to 64 or 128 maybe result in crash with aocl-blis
    public static int BlockSize => 128;

    public static void HHStepBlock(MatrixView Work, MatrixView TU, MatrixView TV, 
        VectorView d, VectorView e, VectorView buffer)
    {
        var iStart = 0;
        while (iStart < Work.Cols)
        {
            var block = Math.Min(BlockSize, Work.Cols - iStart);
            var iEnd = iStart + block;
            var AB1 = Work[iStart.., iStart..iEnd];
            var A1B = Work[iStart..iEnd, iStart..];
            var TUT1 = TU[..block, iStart..iEnd];
            var TVT1 = iEnd == Work.Cols ?
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
                var right = Work[iStart.., (iStart + 1)..];
                Tridiagonaling.BuildHH(col, out d[iStart], out tau);

                var bufferCurrent = buffer[..(Work.Cols - iStart - 1)];
                // right -= col * (col^T * right) / tau
                col.LeftMul(1 / tau, right, bufferCurrent);
                right.Rank1(-1, col, bufferCurrent);

                a10.CopyTo(t01);
                a21.LeftMul(A20, 1.0, t01);

                if (iStart < Work.Cols - 1)
                {
                    tau = ref TVT1[i, i];
                    t01 = TVT1[..i, i];
                    var row = A1B[i, (i + 1)..];
                    var a01 = A1B[..i, i + 1];
                    var a12 = A1B[i, (i + 2)..];
                    var A02 = A1B[..i, (i + 2)..];
                    var bottom = Work[(iStart + 1).., (iStart + 1)..];
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

    public static void BuildUV(MatrixView Work, MatrixView TU, MatrixView TV, 
        MatrixView U, MatrixView V, int TUBlockSize, int TVBlockSize)
    {
        var wid = Work.Cols;
        var wid2 = wid - 1;
        var Tmp = Matrix.Create(TUBlockSize, wid);

        var A = Work;
        var T = TU;
        var W = Tmp[..TUBlockSize, ..];
        var Q = U;
        //Crash here
        QR.ApplyQlnfc(A, T, W, Q);

        A = Work[..wid2, 1..].T;
        T = TV[.., ..^1];
        W = Tmp[..TVBlockSize, ..wid2];
        Q = V[1.., 1..];
        QR.ApplyQlnfc(A, T, W, Q);
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
}
