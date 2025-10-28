namespace SimpleExample.SymmEVD.Tridiag;

public abstract class TridiagBase
{
    protected MatrixView Work { get; }

    public abstract MatrixView Q { get; }

    public VectorView Diag { get; }

    public VectorView SubDiag { get; }

    public abstract void Kernel();

    public Matrix GetTriMatrix()
    {
        var len = Diag.Length;
        Matrix res = Matrix.Create(len, len);
        int i = 0;
        for (; i < len - 1; i++)
        {
            res[i, i] = Diag[i];
            res[i, i + 1] = SubDiag[i];
            res[i + 1, i] = SubDiag[i];
        }
        res[i, i] = Diag[i];
        return res;
    }

    public TridiagBase(Matrix orig)
    {
        var len = orig.Rows;
        ArgumentOutOfRangeException
            .ThrowIfNotEqual(len, orig.Cols,
            nameof(orig));

        Work = orig.Clone();
        Diag = Vector.Create(len);
        SubDiag = Vector.Create(len - 1);
    }

}
