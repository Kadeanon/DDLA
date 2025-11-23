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

    public static void CheckExpectedResults(MatrixView A)
    {
        double alpha = 1.3;
        int rows = A.Rows, cols = A.Cols;
        var x = CreateVectorRandom(rows);
        var y = CreateVectorRandom(cols);
        var expected = CopyMatrix(A);
        BlasProvider.GeR(alpha, x, y, A);
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = 0; j < A.Cols; j++)
            {
                expected[i, j] += alpha * x[i] * y[j];
            }
        }
        var diff = A - expected;
        double err = BlasProvider.NrmF(diff) / Math.Sqrt(A.Size);
        Assert.AreEqual(0, err, 1e-15, $"Result A mismatch");
    }
}
