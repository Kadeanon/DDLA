using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

[TestClass]
public class TestGeR
{
    internal static int rows = Random.Shared.Next(256, 1024);
    internal static int cols = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorGeR()
    {
        var mat = CreateMatrixRandom(rows, cols);
        CheckExpectedResults(mat);
    }

    [TestMethod]
    public void TestColMajorGeR()
    {
        var mat = CreateMatrixTransRandom(rows, cols);
        CheckExpectedResults(mat);
    }

    [TestMethod]
    public void TestSimpleGeR()
    {
        var mat = CreateMatrixStrideRandom(rows, cols, 15);
        CheckExpectedResults(mat);
    }

    public static void CheckExpectedResults(MatrixView a)
    {
        double alpha = 1.3;
        int rows = a.Rows, cols = a.Cols;
        var x = CreateVectorRandom(rows);
        var y = CreateVectorRandom(cols);
        var result = CopyMatrix(a);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Cols; j++)
            {
                result[i, j] += alpha * x[i] * y[j];
            }
        }
        BlasProvider.GeR(alpha, x, y, a);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Cols; j++)
            {
                Assert.AreEqual(result[i, j], a[i, j], 1e-10, $"Value ({i}, {j}) mismatch");
            }
        }
    }
}
