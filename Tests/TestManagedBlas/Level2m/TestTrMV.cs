using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

[TestClass]
public class TestTrMV
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpTrMV()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpTrMV()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpTrMV()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLoTrMV()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLoTrMV()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestSimpleLoTrMV()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestRowMajorUpTrMVDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorUpTrMVDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestSimpleUpTrMVDiag()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorLoTrMVDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorLoTrMVDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestSimpleLoTrMVDiag()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat, DiagType.Unit);
    }

    public static void CheckExpectedResults(UpLo uplo, 
        MatrixView A, DiagType diag = DiagType.NonUnit)
    {
        var trans = TransType.NoTrans;
        double alpha = 2.0;
        int length = A.Cols;
        var x = CreateVectorRandom(length);
        var x0 = x.Clone();
        var y = CreateVector(length);
        BlasProvider.TrMV(uplo, trans, diag, alpha, A, x);
        if (diag == DiagType.Unit) A.Diag.Fill(1.0);
        BlasProvider.MakeTr(A, uplo);
        BlasProvider.GeMV(alpha, A, x0, 0.0, y);
        double err = BlasProvider.RMS(x - y);
        Assert.AreEqual(0, err, 1e-12, $"Result y mismatch");
    }
}
