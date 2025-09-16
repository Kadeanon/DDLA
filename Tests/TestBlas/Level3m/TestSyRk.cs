namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestSyRk
{
    internal static int size = Random.Shared.Next(256, 1024);
    internal static int reds = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpperSyRk()
    {
        var left =
            CreateMatrixRandom(size, reds);
        var result =
            CreateMatrixRandom(size, size);
        CheckExpectedResults(UpLo.Upper, left, result);
    }

    [TestMethod]
    public void TestColMajorUpperSyRk()
    {
        var left =
            CreateMatrixTransRandom(size, reds);
        var result =
            CreateMatrixTransRandom(size, size);
        CheckExpectedResults(UpLo.Upper, left, result);
    }

    [TestMethod]
    public void TestSimpleUpperSyRk()
    {
        int colStride = 4;
        var left =
            CreateMatrixStrideRandom(size, reds, colStride);
        var result =
            CreateMatrixStrideRandom(size, size, colStride);
        CheckExpectedResults(UpLo.Upper, left, result);
    }

    [TestMethod]
    public void TestRowMajorLowerSyRk()
    {
        var left =
            CreateMatrixRandom(size, reds);
        var result =
            CreateMatrixRandom(size, size);
        CheckExpectedResults(UpLo.Lower, left, result);
    }

    [TestMethod]
    public void TestColMajorLowerSyRk()
    {
        var left =
            CreateMatrixTransRandom(size, reds);
        var result =
            CreateMatrixTransRandom(size, size);
        CheckExpectedResults(UpLo.Lower, left, result);
    }

    [TestMethod]
    public void TestSimpleLowerSyRk()
    {
        int colStride = 4;
        var left =
            CreateMatrixStrideRandom(size, reds, colStride);
        var result =
            CreateMatrixStrideRandom(size, size, colStride);
        CheckExpectedResults(UpLo.Lower, left, result);
    }

    public static void CheckExpectedResults
        (UpLo uplo, MatrixView left, MatrixView result)
    {
        double alpha = 2.0;
        double beta = 1.2;
        var expected = CopyMatrix(result);
        int size = left.Rows, red = left.Cols;
        for (int i = 0; i < size; i++)
        {
            foreach (int j in
                uplo == UpLo.Upper ?
                Enumerable.Range(i, size - i) :
                Enumerable.Range(0, i + 1))
            {
                var sum = 0.0;
                for (int k = 0; k < reds; k++)
                {
                    sum += left[i, k] * left[j, k];
                }
                expected[i, j] = alpha * sum + expected[i, j] * beta;
            }
        }
        BlasProvider.SyRk(uplo, alpha, left, beta, result);
        var diff = expected - result;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                Assert.AreEqual(expected[i, j], result[i, j], 1e-10, $"Value ({i}, {j}) mismatch");
            }
        }
    }

}

