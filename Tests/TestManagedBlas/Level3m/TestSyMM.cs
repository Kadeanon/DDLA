using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level3m;

[TestClass]
public class TestSyMM
{
    internal static int rows = Random.Shared.Next(513, 2048);
    internal static int cols = Random.Shared.Next(513, 2048);

    [TestMethod]
    public void TestUpperSyMMLeft()
    {
        CheckExpectedResults
            (UpLo.Upper, SideType.Left);
    }

    [TestMethod]
    public void TestLowerSyMMLeft()
    {
        CheckExpectedResults
            (UpLo.Lower, SideType.Left);
    }

    [TestMethod]
    public void TestUpperSyMMRight()
    {
        CheckExpectedResults
            (UpLo.Upper, SideType.Right);
    }

    [TestMethod]
    public void TestLowerSyMMRight()
    {
        CheckExpectedResults(UpLo.Lower, SideType.Right);
    }

    public static void CheckExpectedResults(UpLo uplo, SideType side)
    {
        int large = 768;
        int small = 64;
        CheckExpectedResults
            (uplo, side, small, small);
        CheckExpectedResults
            (uplo, side, large, small);
        CheckExpectedResults
            (uplo, side, small, large);
        CheckExpectedResults
            (uplo, side, large, large);
    }
    public static void CheckExpectedResults(UpLo uplo, SideType side, int m, int n)
    {
        double alpha = 1 / Math.PI;
        double beta = Math.E;
        int k = side == SideType.Left ? m : n;
        var A = CreateMatrixRandom(k, k);
        var B = CreateMatrixRandom(m, n);
        var C = CreateMatrixRandom(m, n);
        var CExpected = C.Clone();
        BlasProvider.SyMM(side, uplo, alpha, A, B, beta, C);
        BlisProvider.SyMM(side, uplo, alpha, A, B, beta, CExpected);
        var diff = CExpected - C;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5, $"C mismatch in {m}x{n} state.");
    }
}
