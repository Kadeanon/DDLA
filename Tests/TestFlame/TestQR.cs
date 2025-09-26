using DDLA.Factorizations;
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
            MatrixView a = CreateMatrixRandom(m, n);
            MatrixView orig = CreateMatrixRandom(n, nrhs);
            MatrixView b = a * orig;
            var qr = new QR(a);
            var result = qr.Solve(b);
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
            MatrixView orig = CreateMatrixRandom(m, n);
            var (Q, R) = new QR(orig);
            var result = Q * R;
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
