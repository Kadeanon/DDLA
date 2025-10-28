using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level1m;

[TestClass]
public class TestScal2m
{
    internal static int rows = Random.Shared.Next(40, 96);
    internal static int cols = Random.Shared.Next(33, 77);
    internal static int size = Random.Shared.Next(40, 80);
    internal const int stride = 3;

    // Dense + NoTrans
    [TestMethod]
    public void TestDenseNoTransRowMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans, 2.0,
    CreateMatrixRandom(rows, cols), CreateMatrixRandom(rows, cols));
    [TestMethod]
    public void TestDenseNoTransColMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans, 2.0,
    CreateMatrixTransRandom(rows, cols), CreateMatrixTransRandom(rows, cols));
    [TestMethod]
    public void TestDenseNoTransStride() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans, 2.0,
    CreateMatrixStrideRandom(rows, cols, stride), CreateMatrixStrideRandom(rows, cols, stride + 1));

    // Dense + Trans
    [TestMethod]
    public void TestDenseTransRowMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.OnlyTrans, -0.5,
    CreateMatrixRandom(cols, rows), CreateMatrixRandom(rows, cols));
    [TestMethod]
    public void TestDenseTransColMajor() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.OnlyTrans, -0.5,
    CreateMatrixTransRandom(cols, rows), CreateMatrixTransRandom(rows, cols));
    [TestMethod]
    public void TestDenseTransStride() => RunCase(DiagType.NonUnit, UpLo.Dense, TransType.OnlyTrans, -0.5,
    CreateMatrixStrideRandom(cols, rows, stride), CreateMatrixStrideRandom(rows, cols, stride + 1));

    // Upper/Lower NonUnit
    [TestMethod]
    public void TestUpperNonUnitRowMajor() => RunCase(DiagType.NonUnit, UpLo.Upper, TransType.NoTrans, Math.PI,
    CreateMatrixRandom(size, size), CreateMatrixRandom(size, size));
    [TestMethod]
    public void TestUpperNonUnitColMajor() => RunCase(DiagType.NonUnit, UpLo.Upper, TransType.NoTrans, Math.PI,
    CreateMatrixTransRandom(size, size), CreateMatrixTransRandom(size, size));
    [TestMethod]
    public void TestUpperNonUnitStride() => RunCase(DiagType.NonUnit, UpLo.Upper, TransType.NoTrans, Math.PI,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    [TestMethod]
    public void TestLowerNonUnitRowMajor() => RunCase(DiagType.NonUnit, UpLo.Lower, TransType.NoTrans, Math.PI,
    CreateMatrixRandom(size, size), CreateMatrixRandom(size, size));
    [TestMethod]
    public void TestLowerNonUnitColMajor() => RunCase(DiagType.NonUnit, UpLo.Lower, TransType.NoTrans, Math.PI,
    CreateMatrixTransRandom(size, size), CreateMatrixTransRandom(size, size));
    [TestMethod]
    public void TestLowerNonUnitStride() => RunCase(DiagType.NonUnit, UpLo.Lower, TransType.NoTrans, Math.PI,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    // Upper/Lower Unit
    [TestMethod]
    public void TestUpperUnitRowMajor() => RunCase(DiagType.Unit, UpLo.Upper, TransType.NoTrans, Math.E,
    CreateMatrixRandom(size, size), CreateMatrixRandom(size, size));
    [TestMethod]
    public void TestUpperUnitColMajor() => RunCase(DiagType.Unit, UpLo.Upper, TransType.NoTrans, Math.E,
    CreateMatrixTransRandom(size, size), CreateMatrixTransRandom(size, size));
    [TestMethod]
    public void TestUpperUnitStride() => RunCase(DiagType.Unit, UpLo.Upper, TransType.NoTrans, Math.E,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    [TestMethod]
    public void TestLowerUnitRowMajor() => RunCase(DiagType.Unit, UpLo.Lower, TransType.NoTrans, Math.E,
    CreateMatrixRandom(size, size), CreateMatrixRandom(size, size));
    [TestMethod]
    public void TestLowerUnitColMajor() => RunCase(DiagType.Unit, UpLo.Lower, TransType.NoTrans, Math.E,
    CreateMatrixTransRandom(size, size), CreateMatrixTransRandom(size, size));
    [TestMethod]
    public void TestLowerUnitStride() => RunCase(DiagType.Unit, UpLo.Lower, TransType.NoTrans, Math.E,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    // Diagonal only (Zeros)
    [TestMethod]
    public void TestZerosRowMajor() => RunCase(DiagType.NonUnit, UpLo.Zeros, TransType.NoTrans, 2.25,
    CreateMatrixRandom(size, size), CreateMatrixRandom(size, size));
    [TestMethod]
    public void TestZerosColMajor() => RunCase(DiagType.NonUnit, UpLo.Zeros, TransType.NoTrans, 2.25,
    CreateMatrixTransRandom(size, size), CreateMatrixTransRandom(size, size));
    [TestMethod]
    public void TestZerosStride() => RunCase(DiagType.NonUnit, UpLo.Zeros, TransType.NoTrans, 2.25,
    CreateMatrixStrideRandom(size, size, stride), CreateMatrixStrideRandom(size, size, stride + 1));

    private static void RunCase(DiagType aDiag, UpLo aUplo, TransType aTrans, double alpha, MatrixView A, MatrixView BIn)
    {
        var Bm = CopyMatrix(BIn);
        var Be = CopyMatrix(BIn);
        BlasProvider.Scal2(aDiag, aUplo, aTrans, alpha, A, Bm);
        BlisProvider.Scal2(aDiag, aUplo, aTrans, alpha, A, Be);
        var diff = Be - Bm;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5);
    }
}
