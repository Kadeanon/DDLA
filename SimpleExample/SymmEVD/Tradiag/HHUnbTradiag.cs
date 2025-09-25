using DDLA.BLAS;
using DDLA.Misc.Flags;

using static DDLA.BLAS.BlasProvider;

namespace SimpleExample.SymmEVD.Tradiag;

public class HHUnbTradiag: TridiagBase
{
    public override MatrixView Q { get; }

    public VectorView Taus { get; }

    public HHUnbTradiag(Matrix orig) : base(orig)
    {
        var len = orig.Rows;

        Q = Matrix.Eyes(len).Transpose();
        Taus = Vector.Create(len - 1);
    }

    public override void Kernel()
    {
        int i = 0;
        while (i < Work.Rows - 1)
        {
            Step(i);
            BuildQ(i);
            i++;
        }
        Diag[i] = Work[i, i];
    }

    public void Step(int i)
    {
        Diag[i] = Work[i, i];
        var a21 = Work[(i + 1).., i];
        BuildHH(a21, out SubDiag[i], out Taus[i]);
        ApplyRight(i);
    }

    public void ApplyRight(int i)
    {
        var u = Work[(i + 1).., i];
        var A = Work[(i + 1).., (i + 1)..];
        var tau = Taus[i];
        // w.Length = u.Length = size - i - 1
        VectorView w = Diag[(i + 1)..];
        SyMV(UpLo.Lower, 1, A, u, 0, w);
        var beta = w * u / 2;
        Axpy(-beta / tau, u, w);
        SyR2(UpLo.Lower, -1 / tau, u, w, A);
    }

    public void BuildQ(int i)
    {
        var u = Work[(i + 1).., i];
        var H = Q[.., (i + 1)..];
        var tau = Taus[i];
        VectorView v = Work[0, ..];
        GeMV(1.0, H, u, 0.0, v);
        GeR(-1 / tau, v, u, H);
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

        BlasProvider.InvScal(scale, xLast);
        double scaledLenLast = lenLast / Math.Abs(scale);

        chi = 1;
        sigma = -neg_alpha;
        tau = (1 + scaledLenLast * scaledLenLast) / 2;
    }
}
