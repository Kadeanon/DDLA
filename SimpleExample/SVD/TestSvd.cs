using DDLA.Transformations;
using MKLNET;
using SimpleExample.SVD.Bidiag;
using SimpleExample.SVD.Diag;

namespace SimpleExample.SVD;

public class TestSvd
{
    internal static void Test(int rows, int cols)
    {
        for (int i = 0; i < 20; i++)
        {
            Console.WriteLine($"--- Iteration {i + 1} ---");
            // TestHHUTBidiag(rows, cols);
            TestHHUTBidiag(rows, cols);
            Console.WriteLine();
        }
    }

    static void TestHHUnbBidiag(int rows, int cols)
    {
        var mat = Matrix.RandomDense(rows, cols);
        var orig = mat.Clone();

        DateTime start = DateTime.Now;
        BidiagBase bidiag = new HHUnbBidiag(mat);
        Console.WriteLine($"Create svd with {rows}x{cols}");
        bidiag.Kernel();
        var span = DateTime.Now - start;
        var U = bidiag.U;
        var V = bidiag.V;
        var B = bidiag.GetBiMatrix();
        var diff = U * (B * V.T) - mat;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        Console.WriteLine($"Bidiag time out: {span}");

        var E = mat.EmptyLike().View;
        double maxSv, minSv;

        //start = DateTime.Now;
        //var franc = new FrancisQR(bidiag.Diag, bidiag.SubDiag,
        //    bidiag.U, bidiag.V);
        //franc.Kernel();
        //span = DateTime.Now - start;

        //var svdValues = bidiag.Diag;

        //E.Diag = svdValues;
        //diff = U * (E * V.T) - mat;
        //Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        //Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        //Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        //Console.WriteLine($"Diag time out: {span}");

        //maxSv = svdValues[0];
        //minSv = svdValues[^1];
        //Console.WriteLine($"Max sv {maxSv}, min sv {minSv}, cond {maxSv / minSv}");

        var toMkl = orig.Clone();
        var s = Vector.Create(cols);
        var UMKL = Matrix.Create(rows, rows);
        var VMKL = Matrix.Create(cols, cols);
        start = DateTime.Now;
        MKL.set_threading_layer(MklThreading.TBB);
        Lapack.gesdd(Layout.RowMajor, 'A',
            rows, cols, toMkl.Data, toMkl.RowStride,
            s.Data,
            UMKL.Data, rows,
            VMKL.Data, cols);
        span = DateTime.Now - start;

        E.Diag = s;
        diff = UMKL * E * VMKL - mat;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(UMKL)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(VMKL)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");
        Console.WriteLine($"MKL time out: {span}");

        maxSv = s[0];
        minSv = s[^1];
        Console.WriteLine($"Max sv {maxSv}, min sv {minSv}, cond {maxSv / minSv}");
    }

    static void TestHHUTBidiag(int rows, int cols)
    {
        var mat = Matrix.RandomDense(rows, cols);
        var orig = mat.Clone();

        DateTime start = DateTime.Now;
        BidiagBase bidiag = new HHUTBidiag(mat);
        Console.WriteLine($"Create svd with {rows}x{cols}");
        bidiag.Kernel();
        var span = DateTime.Now - start;
        var U = bidiag.U;
        var V = bidiag.V;
        var B = bidiag.GetBiMatrix();
        var diff = U * (B * V.T) - mat;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        Console.WriteLine($"Bidiag time out: {span}");

        var E = mat.EmptyLike().View;
        double maxSv, minSv;

        start = DateTime.Now;
        var franc = new FrancisQRSVD(bidiag.Diag, bidiag.SubDiag,
            bidiag.U, bidiag.V);
        franc.Kernel();
        span = DateTime.Now - start;

        var svdValues = bidiag.Diag;

        E.Diag = svdValues;
        diff = U * (E * V.T) - mat;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        Console.WriteLine($"Diag time out: {span}");

        maxSv = svdValues[0];
        minSv = svdValues[^1];
        Console.WriteLine($"Max sv {maxSv}, min sv {minSv}, cond {maxSv / minSv}");

        var toMkl = orig.Clone();
        var s = Vector.Create(cols);
        var UMKL = Matrix.Create(rows, rows);
        var VMKL = Matrix.Create(cols, cols);
        start = DateTime.Now;
        MKL.set_threading_layer(MklThreading.TBB);
        Lapack.gesdd(Layout.RowMajor, 'A',
            rows, cols, toMkl.Data, toMkl.RowStride,
            s.Data,
            UMKL.Data, rows,
            VMKL.Data, cols);
        span = DateTime.Now - start;

        E.Diag = s;
        diff = UMKL * E * VMKL - mat;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(UMKL)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(VMKL)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");
        Console.WriteLine($"MKL time out: {span}");

        maxSv = s[0];
        minSv = s[^1];
        Console.WriteLine($"Max sv {maxSv}, min sv {minSv}, cond {maxSv / minSv}");
    }

    static void TestHHUTBidiag2(int rows, int cols)
    {
        var A = Matrix.RandomDense(rows, cols);
        var orig = A.Clone();

        DateTime start = DateTime.Now;
        Bidiagonaling.Bidiag(A, out var U, out var V, out var d, out var e);
        Console.WriteLine($"Create svd with {rows}x{cols}");
        var span = DateTime.Now - start;
        var B = Bidiagonaling.GetBiMatrix(A, d, e);
        var diff = U * (B * V.T) - orig;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        Console.WriteLine($"Bidiag time out: {span}");

        var E = A.EmptyLike().View;
        double maxSv, minSv;

        start = DateTime.Now;
        var franc = new FrancisQRSVD(d, e,
            U, V);
        franc.Kernel();
        span = DateTime.Now - start;

        var svdValues = d;

        E.Diag = svdValues;
        diff = U * (E * V.T) - orig;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        Console.WriteLine($"Diag time out: {span}");

        maxSv = svdValues[0];
        minSv = svdValues[^1];
        Console.WriteLine($"Max sv {maxSv}, min sv {minSv}, cond {maxSv / minSv}");

        var toMkl = orig.Clone();
        var s = Vector.Create(cols);
        var UMKL = Matrix.Create(rows, rows);
        var VMKL = Matrix.Create(cols, cols);
        start = DateTime.Now;
        MKL.set_threading_layer(MklThreading.TBB);
        Lapack.gesdd(Layout.RowMajor, 'A',
            rows, cols, toMkl.Data, toMkl.RowStride,
            s.Data,
            UMKL.Data, rows,
            VMKL.Data, cols);
        span = DateTime.Now - start;

        E.Diag = s;
        diff = UMKL * E * VMKL - orig;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(UMKL)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(VMKL)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");
        Console.WriteLine($"MKL time out: {span}");

        maxSv = s[0];
        minSv = s[^1];
        Console.WriteLine($"Max sv {maxSv}, min sv {minSv}, cond {maxSv / minSv}");
    }

    static void TestTwoStageBidiag(int rows, int cols)
    {
        var mat = Matrix.RandomDense(rows, cols);

        DateTime start = DateTime.Now;
        BidiagBase bidiag = new TwoStageBidiag(mat);
        Console.WriteLine($"Create svd with {rows}x{cols}");
        bidiag.Kernel();
        var span = DateTime.Now - start;
        var U = bidiag.U;
        var V = bidiag.V;
        var B = bidiag.GetBiMatrix();
        var diff = U * (B * V.T) - mat;
        Console.WriteLine($"nrmf(UU^T-I)={UVNorm(U)}");
        Console.WriteLine($"nrmf(VV^T-I)={UVNorm(V)}");
        Console.WriteLine($"nrmf(diff)={diff.NrmF()}");

        Console.WriteLine($"Bidiag time out: {span}");
    }

    static double UVNorm(MatrixView UorV)
    {
        var I = Matrix.Eyes(UorV.Rows);
        var diff = UorV * UorV.T - I;
        return diff.NrmF();
    }
}
