using MKLNET;
using SimpleExample.SymmEVD.Diag;
using SimpleExample.SymmEVD.Tridiag;

namespace SimpleExample.SymmEVD;

internal class TestSEvd
{
    internal static void Test(int len)
    {
        for (int i = 0; i < 20; i++)
        {
            Console.WriteLine($"--- Iteration {i + 1} ---");
            TestHHUTTridiag(len);
            Console.WriteLine();
        }
    }

    static void TestHHUnbTridiag(int len)
    {
        DateTime start = DateTime.Now;
        var mat = Matrix.RandomSPD(len);
        var orig = mat.Clone();

        TridiagBase tridiag = new HHUnbTridiag(mat);
        tridiag.Kernel();
        var Q = tridiag.Q;
        var trans = Q.EmptyLike().T;
        Q.CopyTo(trans);
        Q = trans;
        var T = tridiag.GetTriMatrix();

        Console.WriteLine($"nrmf(diff)={(Q * (T * Q.T) - mat).NrmF()}");
        Console.WriteLine($"nrmf(QQT-I)={(Q * Q.T - Matrix.Eyes(len)).NrmF()}");

        var span = DateTime.Now - start;
        Console.WriteLine($"Tridiag time out: {span}");

        start = DateTime.Now;
        var diag = tridiag.Diag;
        var fran = new FrancisQRSEVD(diag,
            tridiag.SubDiag, Q);
        fran.Kernel();
        var eigenValues = diag;

        span = DateTime.Now - start;
        Console.WriteLine($"Final diag time out: {span}");

        for (var i = 0; i < len; i++)
        {
            var eigenValue = eigenValues[i];
            var eigenVector = Q.GetColumn(i);
            var diffMax = (orig * eigenVector - eigenValue * eigenVector).MaxAbs();
            if (diffMax > 1e-12)
            {
                Console.WriteLine($"[WARNING]Index {i}: eigenvalue {eigenValue}, max residual {diffMax}");
            }
        }
    }

    static void TestHHUTTridiag(int len)
    {
        DateTime start = DateTime.Now;
        var mat = Matrix.RandomSPD(len);
        var orig = mat.Clone();

        TridiagBase tridiag = new HHUTTridiag(mat);
        tridiag.Kernel();
        var Q = tridiag.Q;
        var trans = Q.EmptyLike().T;
        Q.CopyTo(trans);
        Q = trans;
        var T = tridiag.GetTriMatrix();

        Console.WriteLine($"nrmf(diff)={(Q * (T * Q.T) - mat).NrmF()}");
        Console.WriteLine($"nrmf(QQT-I)={(Q * Q.T - Matrix.Eyes(len)).NrmF()}");

        var tridiagSpan = DateTime.Now - start;
        Console.WriteLine($"Tridiag time out: {tridiagSpan}");

        start = DateTime.Now;
        var diag = tridiag.Diag;
        var fran = new FrancisQRSEVD(diag,
            tridiag.SubDiag, Q);
        fran.Kernel();
        var eigenValues = diag;

        var diagSpan = DateTime.Now - start;
        Console.WriteLine($"Final diag time out: {diagSpan}");
        var franSpan = diagSpan + tridiagSpan;
        Console.WriteLine($"Total time out: {franSpan}");

        for (var i = 0; i < len; i++)
        {
            var eigenValue = eigenValues[i];
            var eigenVector = Q.GetColumn(i);
            var diffMax = (orig * eigenVector - eigenValue * eigenVector).MaxAbs();
            if (diffMax > 1e-12)
            {
                Console.WriteLine($"[WARNING]Index {i}: eigenvalue {eigenValue}, max residual {diffMax}");
            }
        }

        var toMkl = orig.Clone();
        var s = Vector.Create(len);
        start = DateTime.Now;
        Lapack.syevd(Layout.RowMajor, 'V', UpLoChar.Lower,
            len, toMkl.Data, toMkl.RowStride,
            s.Data);
        var mklSpan = DateTime.Now - start;

        var E = Matrix.Eyes(len);
        E.Diag = s;
        Q = toMkl;
        var diff = Q * (E * Q.T) - mat;
        Console.WriteLine($"nrmf(QQ^T-I)={QNorm(Q)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");
        Console.WriteLine($"MKL time out: {mklSpan}");
        var speedRate = franSpan.TotalSeconds / mklSpan.TotalSeconds - 1;
        Console.WriteLine($"Speed rate (our / MKL - 1): {speedRate}");
    }

    static double QNorm(MatrixView Q)
    {
        var I = Matrix.Eyes(Q.Rows);
        var diff = Q * Q.T - I;
        return diff.NrmF();
    }
}
