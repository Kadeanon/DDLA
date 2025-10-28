using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level3m;

[TestClass]
public class TestGeMM
{
    internal static int rows = Random.Shared.Next(513, 2048);
    internal static int cols = Random.Shared.Next(513, 2048);
    internal static int reds = Random.Shared.Next(513, 2048);
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
        var result = CreateMatrixTransRandom(rows, cols);
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
        var expected = CopyMatrix(result);
        int rows = left.Rows, cols = right.Cols;
        BlisProvider.GeMM(alpha, left, right, beta, expected);
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

