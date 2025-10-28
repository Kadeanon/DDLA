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
        var mat = CreateMatrixTransRandom
            (75, 75);
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

    public static void CheckExpectedResults
        (UpLo uplo, MatrixView a)
    {
        double alpha = 1.0;
        int length = a.Cols;
        var x = CreateVectorRandom(length);
        for (int i = 0; i < length; i++)
        {
            a[i, i] = (0.5 + length) * (a[i, i] + 1);
        }
        var y = CopyVector(x);
        BlasProvider.TrMV(uplo, alpha, a, x);
        BlasProvider.TrSV(uplo, alpha, a, x);
        BlasProvider.Scal(alpha, x);
        double err = 0.0;

        for (int i = 0; i < length; i++)
        {
            var exp = y[i];
            var act = x[i];
            var err2 = (exp - act) / exp;
            err += err2 * err2;
            Console.WriteLine(
                $"""
                i = {i}
                exp = {exp}, act = {act}, err = {err2}, sumed err = {err}
                """);
        }
        err /= length;
        err = Math.Sqrt(err);
        Assert.AreEqual(0, err, 2e-5, $"Solved x mismatch");
    }
}
