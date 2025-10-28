using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level1m;

[TestClass]
public class TestCopym
{
    internal static int rows = Random.Shared.Next(40, 96);
    internal static int cols = Random.Shared.Next(33, 77);
    internal static int size = Random.Shared.Next(40, 80);
    internal const int stride = 3;

    // Dense + NoTrans
    [TestMethod]
    public void TestDenseNoTransRowMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans,
    CreateMatrixRandom(rows, cols), CreateMatrix(rows, cols));
    [TestMethod]
    public void TestDenseNoTransColMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans,
    CreateMatrixTransRandom(rows, cols), CreateMatrixTrans(rows, cols));
    [TestMethod]
    public void TestDenseNoTransStride() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans,
    CreateMatrixStrideRandom(rows, cols, stride), CreateMatrixStrideRandom(rows, cols, stride + 1));

    // Dense + Trans
    [TestMethod]
    public void TestDenseTransRowMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.OnlyTrans,
    CreateMatrixRandom(cols, rows), CreateMatrix(rows, cols));
    [TestMethod]
    public void TestDenseTransColMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.OnlyTrans,
    CreateMatrixTransRandom(cols, rows), CreateMatrixTrans(rows, cols));
    [TestMethod]
    public void TestDenseTransStride() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.OnlyTrans,
    CreateMatrixStrideRandom(cols, rows, stride), CreateMatrixStrideRandom(rows, cols, stride + 1));

    // Upper/Lower NonUnit
    [TestMethod]
    public void TestUpperNonUnitRowMajor() => RunCase(DiagType.NonUnit, UpLo.Upper, TransType.NoTrans,
    CreateMatrixRandom(size, size), CreateMatrix(size, size));
    [TestMethod]
    public void TestUpperNonUnitColMajor() => RunCase(DiagType.NonUnit, UpLo.Upper, TransType.NoTrans,
    CreateMatrixTransRandom(size, size), CreateMatrixTrans(size, size));
    [TestMethod]
    public void TestUpperNonUnitStride() => RunCase(DiagType.NonUnit, UpLo.Upper, TransType.NoTrans,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    [TestMethod]
    public void TestLowerNonUnitRowMajor() => RunCase(DiagType.NonUnit, UpLo.Lower, TransType.NoTrans,
    CreateMatrixRandom(size, size), CreateMatrix(size, size));
    [TestMethod]
    public void TestLowerNonUnitColMajor() => RunCase(DiagType.NonUnit, UpLo.Lower, TransType.NoTrans,
    CreateMatrixTransRandom(size, size), CreateMatrixTrans(size, size));
    [TestMethod]
    public void TestLowerNonUnitStride() => RunCase(DiagType.NonUnit, UpLo.Lower, TransType.NoTrans,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    // Upper/Lower Unit
    [TestMethod]
    public void TestUpperUnitRowMajor() => RunCase(DiagType.Unit, UpLo.Upper, TransType.NoTrans,
    CreateMatrixRandom(size, size), CreateMatrix(size, size));
    [TestMethod]
    public void TestUpperUnitColMajor() => RunCase(DiagType.Unit, UpLo.Upper, TransType.NoTrans,
    CreateMatrixTransRandom(size, size), CreateMatrixTrans(size, size));
    [TestMethod]
    public void TestUpperUnitStride() => RunCase(DiagType.Unit, UpLo.Upper, TransType.NoTrans,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    [TestMethod]
    public void TestLowerUnitRowMajor() => RunCase(DiagType.Unit, UpLo.Lower, TransType.NoTrans,
    CreateMatrixRandom(size, size), CreateMatrix(size, size));
    [TestMethod]
    public void TestLowerUnitColMajor() => RunCase(DiagType.Unit, UpLo.Lower, TransType.NoTrans,
    CreateMatrixTransRandom(size, size), CreateMatrixTrans(size, size));
    [TestMethod]
    public void TestLowerUnitStride() => RunCase(DiagType.Unit, UpLo.Lower, TransType.NoTrans,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    // Diagonal only (Zeros)
    [TestMethod]
    public void TestZerosRowMajor() => RunCase(DiagType.NonUnit, UpLo.Zeros, TransType.NoTrans,
    CreateMatrixRandom(size, size), CreateMatrix(size, size));
    [TestMethod]
    public void TestZerosColMajor() => RunCase(DiagType.NonUnit, UpLo.Zeros, TransType.NoTrans,
    CreateMatrixTransRandom(size, size), CreateMatrixTrans(size, size));
    [TestMethod]
    public void TestZerosStride() => RunCase(DiagType.NonUnit, UpLo.Zeros, TransType.NoTrans,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    private static void RunCase(DiagType aDiag, UpLo aUplo, TransType aTrans, MatrixView A, Matrix BOut)
    {
        // Fill destination with distinct values to ensure overwritten properly
        BOut.Fill(-123.456);
        var Be = CopyMatrix(BOut);
        BlasProvider.Copy(aDiag, aUplo, aTrans, 0, A, BOut);
        BlisProvider.Copy(aDiag, aUplo, aTrans, A, Be);
        var diff = Be - BOut;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5);
    }
}
