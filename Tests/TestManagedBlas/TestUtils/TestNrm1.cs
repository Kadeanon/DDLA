using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.TestUtils;

[TestClass]
public class TestNrm1
{
    internal static int rows = Random.Shared.Next(48, 96);
    internal static int cols = Random.Shared.Next(33, 77);
    internal static int size = Random.Shared.Next(40, 80);
    internal const int stride = 3;

    // Dense
    [TestMethod] public void TestDenseRowMajor() => RunCase(UpLo.Dense, CreateMatrixRandom(rows, cols));
    [TestMethod] public void TestDenseColMajor() => RunCase(UpLo.Dense, CreateMatrixTransRandom(rows, cols));
    [TestMethod] public void TestDenseStride() => RunCase(UpLo.Dense, CreateMatrixStrideRandom(rows, cols, stride));

    // Upper NonUnit
    [TestMethod] public void TestUpperNonUnitRowMajor() => RunCase(UpLo.Upper, CreateMatrixRandom(size, size));
    [TestMethod] public void TestUpperNonUnitColMajor() => RunCase(UpLo.Upper, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestUpperNonUnitStride() => RunCase(UpLo.Upper, CreateMatrixStrideRandom(size, size, stride));

    // Lower NonUnit
    [TestMethod] public void TestLowerNonUnitRowMajor() => RunCase(UpLo.Lower, CreateMatrixRandom(size, size));
    [TestMethod] public void TestLowerNonUnitColMajor() => RunCase(UpLo.Lower, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestLowerNonUnitStride() => RunCase(UpLo.Lower, CreateMatrixStrideRandom(size, size, stride));

    // Upper Unit
    [TestMethod] public void TestUpperUnitRowMajor() => RunCase(UpLo.Upper, CreateMatrixRandom(size, size));
    [TestMethod] public void TestUpperUnitColMajor() => RunCase(UpLo.Upper, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestUpperUnitStride() => RunCase(UpLo.Upper, CreateMatrixStrideRandom(size, size, stride));

    // Lower Unit
    [TestMethod] public void TestLowerUnitRowMajor() => RunCase(UpLo.Lower, CreateMatrixRandom(size, size));
    [TestMethod] public void TestLowerUnitColMajor() => RunCase(UpLo.Lower, CreateMatrixTransRandom(size, size));
    [TestMethod] public void TestLowerUnitStride() => RunCase(UpLo.Lower, CreateMatrixStrideRandom(size, size, stride));

    private static void RunCase(UpLo uplo, MatrixView A)
    {
        var blas_nrm1 = BlasProvider.NrmF(A, uplo);
        var blis_nrm1 = BlisProvider.NrmF(A, uplo);
        Assert.AreEqual(blis_nrm1, blas_nrm1, 1e-10);
    }
}
