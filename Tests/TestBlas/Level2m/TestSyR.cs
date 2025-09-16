namespace Tests.TestBlas.Level2m;

[TestClass]
public class TestSyR
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpSyR()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpSyR()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpSyR()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLoSyR()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLoSyR()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestSimpleLoSyR()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    public static void CheckExpectedResults(UpLo uplo, MatrixView a)
    {
        double alpha = 2.0;
        var length = a.Rows;
        var x = CreateVectorRandom(length);
        bool upper = uplo == UpLo.Upper;
        var y = CopyMatrix(a);
        BlasProvider.SyR(uplo, alpha, x, a);
        if (upper)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = i; j < a.Cols; j++)
                {
                    double expected = y[i, j] + alpha * x[i] * x[j];
                    Assert.AreEqual(expected, a[i, j], 1e-10, $"Matrix {i},{j} mismatch");
                }
            }
        }
        else
        {
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double expected = y[i, j] + alpha * x[i] * x[j];
                    Assert.AreEqual(expected, a[i, j], 1e-10, $"Matrix {i},{j} mismatch");
                }
            }
        }
    }
}
