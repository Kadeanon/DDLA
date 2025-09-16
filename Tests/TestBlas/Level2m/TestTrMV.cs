namespace Tests.TestBlas.Level2m;

[TestClass]
public class TestTrMV
{
    internal static int length = Random.Shared.Next(2, 4);//256, 1024);

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

    public static void CheckExpectedResults(UpLo uplo, MatrixView a)
    {
        double alpha = 2.0;
        int length = a.Cols;
        var x = CreateVectorRandom(length);
        var x0 = x.Clone();
        var y = CreateVector(length);
        BlasProvider.MakeTr(a, uplo);
        BlasProvider.GeMV(alpha, a, x0, 0.0, y);
        BlasProvider.TrMV(uplo, alpha, a, x);
        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(y[i], x[i], 1e-10, $"Row {i} mismatch");
        }
    }
}
