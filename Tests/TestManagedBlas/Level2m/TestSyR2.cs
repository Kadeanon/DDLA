using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

[TestClass]
public class TestSyr2
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpSyR2()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpSyR2()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpSyR2()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLoSyR2()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLoSyR2()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestSimpleLoSyR2()
    {
        int colStride = 2;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    public static void CheckExpectedResults(UpLo uplo, MatrixView A)
    {
        double alpha = 2.0;
        int cols = A.Cols, rows = A.Rows;
        var vecx = CreateVectorRandom(rows);
        var vecy = CreateVectorRandom(cols);
        bool upper = uplo == UpLo.Upper;
        var expected = CopyMatrix(A);
        BlasProvider.SyR2(uplo, alpha, vecx, vecy, A);
        BlasProvider.GeR(alpha, vecx, vecy, expected);
        BlasProvider.GeR(alpha, vecy, vecx, expected);
        var diff = A - expected;
        diff.MakeTr(uplo);
        double err = BlasProvider.NrmF(diff) / Math.Sqrt(A.Size);
        Assert.AreEqual(0, err, 1e-14, $"Result A mismatch");
    }
}
