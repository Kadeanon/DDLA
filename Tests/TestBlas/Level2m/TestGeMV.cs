namespace Tests.TestBlas.Level2m;

[TestClass]
public class TestGeMV
{
    internal static int rows = Random.Shared.Next(256, 1024);
    internal static int cols = Random.Shared.Next(256, 1024);
    [TestMethod]
    public void TestRowMajorGeMV()
    {
        var mat = CreateMatrixRandom(rows, cols);
        CheckExpectedResults(mat);
    }

    [TestMethod]
    public void TestColMajorGeMV()
    {
        var mat = CreateMatrixTransRandom(rows, cols);
        CheckExpectedResults(mat);
    }

    [TestMethod]
    public void TestSimpleGeMV()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom(rows, cols, colStride);
        CheckExpectedResults(mat);
    }

    public static void CheckExpectedResults(MatrixView a)
    {
        double alpha = 2.0;
        double beta = 1.2;
        var aTrue = a.Clone();
        int rows = aTrue.Rows, cols = aTrue.Cols;
        var x = CreateVectorRandom(cols);
        var y = CreateVectorRandom(rows);
        var result = CopyVector(y);
        for (int i = 0; i < rows; i++)
        {
            result[i] *= beta;
            for (int j = 0; j < cols; j++)
            {
                result[i] += alpha * aTrue[i, j] * x[j];
            }
        }
        BlasProvider.GeMV(alpha, a, x, beta, y);
        for (int i = 0; i < rows; i++)
        {
            Assert.AreEqual(result[i], y[i], 1e-10, $"Row {i} mismatch");
        }
    }
}
