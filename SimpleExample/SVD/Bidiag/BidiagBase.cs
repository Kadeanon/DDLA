namespace SimpleExample.SVD.Bidiag;

public abstract class BidiagBase
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

    public BidiagBase(Matrix orig)
    {
        var len = orig.Rows;
        var wid = orig.Cols;
        if (len < wid)
            throw new ArgumentException("Only support m >= n");
        Work = orig.Clone(colMajor:true);
        Diag = Vector.Create(wid);
        SubDiag = Vector.Create(wid - 1);
    }
}
