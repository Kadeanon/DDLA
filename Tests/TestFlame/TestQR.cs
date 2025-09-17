using static DDLA.Factorizations.QR;

namespace Tests.TestFlame;

[TestClass]
public class TestQR
{
    private const int tiny = 4;
    private const int small = 64;
    private const int medium = 256;
    private const int large = 1024;

    [TestMethod]
    public void TestTinyUnbQRSquare()
        => TestQRDec(tiny, tiny);

    [TestMethod]
    public void TestSmallUnbQRThin()
        => TestQRDec(small * 3 + 1, small);

    [TestMethod]
    public void TestMediumBloQRSquare()
        => TestQRDec(medium, medium);

    [TestMethod]
    public void TestMediumBloQRThin()
        => TestQRDec(medium * 3 + 1, medium);

    [TestMethod]
    public void TestLargeBloQRSquareSolver()
        => TestQRSolve(large, large, large, 1e-6);

    private static void TestQRSolve(int m, int n, int nrhs, double tol = 1e-10)
    {
        int count = 20;
        for (int i = 0; i < count; i++)
        {
            MatrixView a = CreateUtils.CreateMatrixRandom(m, n);
            MatrixView orig = CreateUtils.CreateMatrixRandom(n, nrhs);
            MatrixView b = a * orig;
            QRDecompose(a, out var t);
            var result = orig.EmptyLike();
            Solve(a, t, b, result);
            var diff = orig - result;
            double n1 = diff.Nrm1(),
                n2 = diff.NrmF(),
                nf = diff.NrmInf();
            Assert.IsTrue(n1 < tol,
                $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol,
                $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol,
                $"NormInf error {nf} exceeds tolerance {tol}.");
            Console.WriteLine($"Test {i + 1}/{count}:" +
                $"n1 = {n1}, n2 = {n2}, nf = {nf}");
        }
    }

    private static void TestQRDec(int m, int n, double tol = 1e-10)
    {
        int count = 20;
        for (int i = 0; i < count; i++)
        {
            MatrixView orig = CreateUtils.CreateMatrixRandom(m, n);
            MatrixView work = CreateUtils.CopyMatrix(orig);
            MatrixView t = CreateT(work);
            QRDecompose(work, t);
            var r = work.Clone().View;
            var diag = r.Diag.Clone();
            BlasProvider.Set(DiagType.NonUnit, UpLo.Lower, 0.0, r);
            r.Diag = diag;
            var q = Matrix.Eyes(m);
            FormQ(work, t, q);
            var result = q * r;
            var diff = orig - result;
            double n1 = diff.Nrm1(),
                n2 = diff.NrmF(),
                nf = diff.NrmInf();
            Assert.IsTrue(n1 < tol,
                $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol,
                $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol,
                $"NormInf error {nf} exceeds tolerance {tol}.");
            Console.WriteLine($"Test {i + 1}/{count}:" +
                $"n1 = {n1}, n2 = {n2}, nf = {nf}");
        }
    }
}
