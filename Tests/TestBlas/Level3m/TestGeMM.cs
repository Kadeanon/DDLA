namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestGeMM
{
    internal static int rows = Random.Shared.Next(64, 1024);
    internal static int cols = Random.Shared.Next(64, 1024);
    internal static int reds = Random.Shared.Next(64, 1024);
    [TestMethod]
    public void TestRowMajorGeMM()
    {
        var left = CreateMatrixRandom(rows, reds);
        var right = CreateMatrixRandom(reds, cols);
        var result = CreateMatrixRandom(rows, cols);
        CheckExpectedResults(left, right, result);
    }

    [TestMethod]
    public void TestColMajorGeMM()
    {
        var left = CreateMatrixTransRandom(rows, reds);
        var right = CreateMatrixTransRandom(reds, cols);
        var result = CreateMatrixTransRandom(rows, cols).Clone();
        CheckExpectedResults(left, right, result);
    }

    [TestMethod]
    public void TestSimpleGeMM()
    {
        int colStride = 4;
        var left = CreateMatrixStrideRandom(rows, reds, colStride);
        var right = CreateMatrixStrideRandom(reds, cols, colStride);
        var result = CreateMatrixStrideRandom(rows, cols, colStride);
        CheckExpectedResults(left, right, result);
    }

    public static void CheckExpectedResults(MatrixView left, MatrixView right, MatrixView result)
    {
        double alpha = 2.0;
        double beta = 1.2;
        var leftTrue = left.Clone();
        var rightTrue = right.Clone();
        var expected = CopyMatrix(result);
        int rows = leftTrue.Rows, cols = rightTrue.Cols, reds = rightTrue.Rows;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                var sum = 0.0;
                for (int k = 0; k < reds; k++)
                {
                    sum += alpha * leftTrue[i, k] * rightTrue[k, j];
                }
                expected[i, j] = sum + expected[i, j] * beta;
            }
        }
        BlasProvider.GeMM(alpha, left, right, beta, result);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Assert.AreEqual(expected[i, j], result[i, j], 1e-10, $"Value ({i}, {j}) mismatch");
            }
        }
    }
}

