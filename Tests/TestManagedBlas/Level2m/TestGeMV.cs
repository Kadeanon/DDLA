using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

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

    public static void CheckExpectedResults(MatrixView A)
    {
        double alpha = 2.0;
        double beta = 1.2;
        int rows = A.Rows, cols = A.Cols;
        var x = CreateVectorRandom(cols);
        var y = CreateVectorRandom(rows);
        var expected = CopyVector(y).Scaled(beta);
        for (int i = 0; i < rows; i++)
        {
            expected[i] += alpha * A[i, ..] * x;
        }
        BlasProvider.GeMV(alpha, A, x, beta, y);
        double err = BlasProvider.RMS(y - expected);
        Assert.AreEqual(0, err, 1e-12, $"Result y mismatch");
    }
}
