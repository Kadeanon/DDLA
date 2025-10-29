using SimpleExample.LAFFExercise.QRs;
using SimpleExample.SymmEVD.Tridiag;

namespace SimpleExample.SVD.Bidiag;

public class HHUTBidiag : BidiagBase
{
    public HHUTBidiag(Matrix orig) : base(orig)
    {
        var len = orig.Rows;
        var wid = orig.Cols;
        U = Matrix.Eyes(len, colMajor: true);
        V = Matrix.Eyes(wid, colMajor: true);
        TU = Matrix.Create(TUBlockSize, wid, colMajor: true);
        TV = Matrix.Create(TVBlockSize, wid);
        Buffer = Vector.Create(len);
    }

    public override MatrixView U { get; }

    public override MatrixView V { get; }

    public MatrixView TU { get; }

    public MatrixView TV { get; }

    public VectorView Buffer { get; }

    // set to 64 or 128 maybe result in crash with aocl-blis
    public static int MaxBlockSize => 128; 

    public int TUBlockSize => Math.Min(MaxBlockSize, Work.Cols);

    public int TVBlockSize => Math.Min(MaxBlockSize, Work.Cols - 1);

    public override void Kernel()
    {
        HHStepBlock();
        BuildUV();
    }

    public void HHStepBlock()
    {
        var iStart = 0;
        while (iStart < Work.Cols)
        {
            var block = Math.Min(MaxBlockSize, Work.Cols - iStart);
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
                HHUnbTridiag.BuildHH(col, out Diag[iStart], out tau);

                var buffer = Buffer[..(Work.Cols - iStart - 1)];
                // right -= col * (col^T * right) / tau
                col.LeftMul(1 / tau, right, buffer);
                right.Rank1(-1, col, buffer);

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
                    HHUnbTridiag.BuildHH(row, out SubDiag[iStart], out tau);
                    buffer = Buffer[..bottom.Rows];
                    // bottom -= (bottom * row) * row^T / tau
                    bottom.Multify(1 / tau, row, buffer);
                    bottom.Rank1(-1, buffer, row);
                    //ApplyUV(row, VRight, tau);

                    a01.CopyTo(t01);
                    A02.Multify(a12, 1.0, t01);
                }
                iStart++;
            }
        }
    }

    public void BuildUV()
    {
        var wid = Work.Cols;
        var wid2 = wid - 1;
        var Tmp = Matrix.Create(TUBlockSize, wid);

        var A = Work;
        var T = TU;
        var W = Tmp[..TUBlockSize, ..];
        var Q = U;
        //Crash here
        HHUTQR.ApplyQ(A, T, W, Q);

        A = Work[..wid2, 1..].T;
        T = TV[.., ..^1];
        W = Tmp[..TVBlockSize, ..wid2];
        Q = V[1.., 1..];
        HHUTQR.ApplyQ(A, T, W, Q);
    }
}
