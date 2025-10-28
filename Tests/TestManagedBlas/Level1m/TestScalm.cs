using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level1m;

[TestClass]
public class TestScalm
{
    internal static int rows = Random.Shared.Next(40, 96);
    internal static int cols = Random.Shared.Next(33, 77);
    internal static int size = Random.Shared.Next(40, 80);
    internal const int stride = 3;

    // Dense
    [TestMethod] public void TestDenseRowMajor() => RunCase(UpLo.Dense, -3.25, CreateMatrixRandom(rows, cols));
    [TestMethod] public void TestDenseColMajor() => RunCase(UpLo.Dense, -3.25, CreateMatrixTransRandom(rows, cols));
    [TestMethod] public void TestDenseStride() => RunCase(UpLo.Dense, -3.25, CreateMatrixStrideRandom(rows, cols, stride));

    // Upper
    [TestMethod] public void TestUpperRowMajor() => RunCase(UpLo.Upper, 0.125, CreateMatrixRandom(size, size));
    [TestMethod] public void TestUpperColMajor() => RunCase(UpLo.Upper, 0.125, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestUpperStride() => RunCase(UpLo.Upper, 0.125, CreateMatrixStrideRandom(size, size, stride));

    // Lower
    [TestMethod] public void TestLowerRowMajor() => RunCase(UpLo.Lower, 2.0, CreateMatrixRandom(size, size));
    [TestMethod] public void TestLowerColMajor() => RunCase(UpLo.Lower, 2.0, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestLowerStride() => RunCase(UpLo.Lower, 2.0, CreateMatrixStrideRandom(size, size, stride));

    private static void RunCase(UpLo aUplo, double alpha, MatrixView Ain)
    {
        var Am = CopyMatrix(Ain);
        var Ae = CopyMatrix(Ain);
        BlasProvider.Scal(aUplo, alpha, Am);
        BlisProvider.Scal(aUplo, alpha, Am);
        var diff = Ae - Am;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5);
    }
}
