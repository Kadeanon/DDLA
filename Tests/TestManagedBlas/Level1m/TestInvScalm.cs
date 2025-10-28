using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level1m;

[TestClass]
public class TestInvScalm
{
    internal static int rows = Random.Shared.Next(40, 96);
    internal static int cols = Random.Shared.Next(33, 77);
    internal static int size = Random.Shared.Next(40, 80);
    internal const int stride = 3;

    // Dense
    [TestMethod] public void TestDenseRowMajor() => RunCase(UpLo.Dense, 2.5, CreateMatrixRandom(rows, cols));
    [TestMethod] public void TestDenseColMajor() => RunCase(UpLo.Dense, 2.5, CreateMatrixTransRandom(rows, cols));
    [TestMethod] public void TestDenseStride() => RunCase(UpLo.Dense, 2.5, CreateMatrixStrideRandom(rows, cols, stride));

    // Upper
    [TestMethod] public void TestUpperRowMajor() => RunCase(UpLo.Upper, Math.E, CreateMatrixRandom(size, size));
    [TestMethod] public void TestUpperColMajor() => RunCase(UpLo.Upper, Math.E, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestUpperStride() => RunCase(UpLo.Upper, Math.E, CreateMatrixStrideRandom(size, size, stride));

    // Lower
    [TestMethod] public void TestLowerRowMajor() => RunCase(UpLo.Lower, Math.PI, CreateMatrixRandom(size, size));
    [TestMethod] public void TestLowerColMajor() => RunCase(UpLo.Lower, Math.PI, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestLowerStride() => RunCase(UpLo.Lower, Math.PI, CreateMatrixStrideRandom(size, size, stride));

    private static void RunCase(UpLo aUplo, double alpha, MatrixView Ain)
    {
        var Am = CopyMatrix(Ain);
        var Ae = CopyMatrix(Ain);
        BlasProvider.InvScal(aUplo, alpha, Am);
        for (int i = 0; i < Ain.Rows; i++)
        {
            for (int j = 0; j < Ain.Cols; j++)
            {
                ref var val = ref Ae[i, j];
                if (aUplo == UpLo.Dense ||
                    (aUplo == UpLo.Upper && j >= i) ||
                    (aUplo == UpLo.Lower && i >= j))
                {
                    val /= alpha;
                }
            }
        }
        var diff = Ae - Am;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5);
    }
}
