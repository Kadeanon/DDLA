
using DDLA.Fatorizations;

namespace Tests.TestFlame;

[TestClass]
public partial class TestLU
{
    #region nopiv

    private const int tiny = 4;
    private const int small = 64;
    private const int medium = 256;
    private const int large = 1024;
    private static int seed {get; } = Random.Shared.Next();

    [TestMethod]
    public void TestTinyUnbLU()
        => TestLUDec(tiny, LU.LUDecUnblock);

    [TestMethod]
    public void TestSmallUnbLU()
        => TestLUDec(small, LU.LUDecUnblock);

    [TestMethod]
    public void TestMediumBloLU()
        => TestLUDec(medium, LU.LUDecBlock);
    #endregion nopiv

    #region nopiv static
    private static void TestLUDec(int length, Action<MatrixView> action, double tol = 1e-6)
    {
        Random random = new(seed);
        var count = 20;
        for (int i = 0; i < count; i++)
        {
            var lower = CreateMatrixRandom
            (length, length, random).View;
            MakeLMat(lower);
            var upper = CreateMatrixRandom
            (length, length, random).View;
            MakeUMat(upper);
            var orig = lower * upper;
            orig.CopyTo(lower);
            action(lower);
            lower.CopyTo(upper);
            MakeLMat(lower);
            MakeUMat(upper);
            var result = lower * upper;
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
    #endregion nopiv static

    private static void MakeLMat(MatrixView lower)
    {
        BlasProvider.MakeTr(lower, UpLo.Lower);
        lower.Diag.Fill(1);
    }

    private static void MakeUMat(MatrixView upper)
    {
        BlasProvider.MakeTr(upper, UpLo.Upper);
    }

}
