namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestSyR2k
{
    internal static int small = 64;
    internal static int reds = 1024;
    
    [TestMethod]
    public void TestUpperSyR2k()
    {
        CheckExpectedResults(UpLo.Upper);
    }

    [TestMethod]
    public void TestLowerSyR2k()
    {
        CheckExpectedResults(UpLo.Lower);
    }

    public static void CheckExpectedResults(UpLo uplo)
    {
        int m = 512;
        int k = 260;

        double alpha = 1 / Math.PI;
        double beta = Math.E;

        var A = CreateMatrixRandom(m, k);
        var B = CreateMatrixRandom(m, k);
        var C = CreateMatrix(m, m);
        BlasProvider.MakeSy(C, uplo);
        var CExpected = C.Clone();
        BlasProvider.SyR2k(uplo, alpha, A, B, beta, C);
        BlasProvider.MakeSy(C, uplo);
        BlasProvider.GeMM(alpha, A, B.Transpose(), beta, CExpected);
        beta = 1.0;
        BlasProvider.GeMM(alpha, B, A.Transpose(), beta, CExpected);
        var diff = CExpected - C;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5, $"C mismatch in {m}x{k} state with uplo {uplo}.");
    }
}
