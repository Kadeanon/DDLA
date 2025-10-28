using SimpleExample.SymmEVD.Tridiag;

namespace SimpleExample.SVD.Bidiag;

public class HHUnbBidiag : BidiagBase
{
    public HHUnbBidiag(Matrix orig): base(orig)
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
            if(i < Work.Cols - 1)
                StepV(i);
        }
    }

    public void StepU(int i)
    {
        var col = Work[i.., i];
        var right = Work[i.., (i + 1)..];
        var URight = U[.., i..];
        HHUnbTridiag.BuildHH(col, out Diag[i], out var tau1);
        ApplyRight(col, right, tau1);
        ApplyUV(col, URight, tau1);
    }

    public void StepV(int i)
    {
        var row = Work[i, (i + 1)..];
        var bottom = Work[(i + 1).., (i + 1)..];
        var VRight = V[.., (i + 1)..];
        HHUnbTridiag.BuildHH(row, out SubDiag[i], out var tau2);
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
