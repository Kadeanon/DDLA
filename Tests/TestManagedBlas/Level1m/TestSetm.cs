using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level1m;

[TestClass]
public class TestSetm
{
    internal static int rows = Random.Shared.Next(40, 96);
    internal static int cols = Random.Shared.Next(33, 77);
    internal static int size = Random.Shared.Next(40, 80);
    internal const int stride = 3;

    // Dense
    [TestMethod] public void TestDenseRowMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, 0.0, CreateMatrixRandom(rows, cols));
    [TestMethod] public void TestDenseColMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, 0.0, CreateMatrixTransRandom(rows, cols));
    [TestMethod] public void TestDenseStride() => RunCase(DiagType.NonUnit, UpLo.Dense, 0.0, CreateMatrixStrideRandom(rows, cols, stride));

    // Upper NonUnit
    [TestMethod] public void TestUpperNonUnitRowMajor() => RunCase(DiagType.NonUnit, UpLo.Upper, 1.25, CreateMatrixRandom(size, size));
    [TestMethod] public void TestUpperNonUnitColMajor() => RunCase(DiagType.NonUnit, UpLo.Upper, 1.25, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestUpperNonUnitStride() => RunCase(DiagType.NonUnit, UpLo.Upper, 1.25, CreateMatrixStrideRandom(size, size, stride));

    // Lower NonUnit
    [TestMethod] public void TestLowerNonUnitRowMajor() => RunCase(DiagType.NonUnit, UpLo.Lower, -1.25, CreateMatrixRandom(size, size));
    [TestMethod] public void TestLowerNonUnitColMajor() => RunCase(DiagType.NonUnit, UpLo.Lower, -1.25, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestLowerNonUnitStride() => RunCase(DiagType.NonUnit, UpLo.Lower, -1.25, CreateMatrixStrideRandom(size, size, stride));

    // Upper Unit
    [TestMethod] public void TestUpperUnitRowMajor() => RunCase(DiagType.Unit, UpLo.Upper, -2.5, CreateMatrixRandom(size, size));
    [TestMethod] public void TestUpperUnitColMajor() => RunCase(DiagType.Unit, UpLo.Upper, -2.5, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestUpperUnitStride() => RunCase(DiagType.Unit, UpLo.Upper, -2.5, CreateMatrixStrideRandom(size, size, stride));

    // Lower Unit
    [TestMethod] public void TestLowerUnitRowMajor() => RunCase(DiagType.Unit, UpLo.Lower, 3.75, CreateMatrixRandom(size, size));
    [TestMethod] public void TestLowerUnitColMajor() => RunCase(DiagType.Unit, UpLo.Lower, 3.75, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestLowerUnitStride() => RunCase(DiagType.Unit, UpLo.Lower, 3.75, CreateMatrixStrideRandom(size, size, stride));

    // Diagonal only (Zeros)
    [TestMethod] public void TestZerosRowMajor() => RunCase(DiagType.NonUnit, UpLo.Zeros, Math.PI, CreateMatrixRandom(size, size));
    [TestMethod] public void TestZerosColMajor() => RunCase(DiagType.NonUnit, UpLo.Zeros, Math.PI, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestZerosStride() => RunCase(DiagType.NonUnit, UpLo.Zeros, Math.PI, CreateMatrixStrideRandom(size, size, stride));

    private static void RunCase(DiagType diag, UpLo aUplo, double alpha, MatrixView Ain)
    {
        var Am = CopyMatrix(Ain);
        var Ae = CopyMatrix(Ain);
        BlasProvider.Set(diag, aUplo, alpha, Am);
        BlisProvider.Set(diag, aUplo, alpha, Ae);
        var diff = Ae - Am;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5);
    }
}
