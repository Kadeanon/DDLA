using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

[TestClass]
public class TestTrSV
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpTrSV()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpTrSV()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpTrSV()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLoTrSV()
    {
        var mat = CreateMatrixRandom(
            length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLoTrSV()
    {
        var mat = CreateMatrixTransRandom(
            length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestSimpleLoTrSV()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestRowMajorUpTrSVDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorUpTrSVDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestSimpleUpTrSVDiag()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorLoTrSVDiag()
    {
        var mat = CreateMatrixRandom(
            length, length);
        CheckExpectedResults(UpLo.Lower, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorLoTrSVDiag()
    {
        var mat = CreateMatrixTransRandom(
            length, length);
        CheckExpectedResults(UpLo.Lower, mat, DiagType.Unit);
    }

    [TestMethod]
    public void TestSimpleLoTrSVDiag()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat, DiagType.Unit);
    }

    public static void CheckExpectedResults
        (UpLo uplo, MatrixView A, DiagType diag = DiagType.NonUnit)
    {
        var trans = TransType.NoTrans;
        double alpha = 2.0;
        int length = A.Cols;
        var x = CreateVectorRandom(length);
        MatrixView.RandomTriangle(A.Rows, uplo).CopyTo(A);
        var ADiag = A.Diag;
        for (int i = 0; i < length; i++)
        {
            ADiag[i] = (0.5 + length) * (ADiag[i] + 1);
        }
        var y = CopyVector(x);
        BlasProvider.TrMV(uplo, trans, diag, alpha, A, x);
        BlasProvider.TrSV(uplo, trans, diag, 1 / alpha, A, x);
        double err = BlasProvider.RMS(x - y);
        Assert.AreEqual(0, err, 1e-14, $"Solved x mismatch");
    }
}
