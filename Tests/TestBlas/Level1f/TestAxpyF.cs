namespace Tests.TestBlas.Level1f;

[TestClass]
public class TestAxpyF
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);
    internal static int axpyFPref = 8;

    [TestMethod]
    public void TestContinueAxpyFLess()
        => TestContinueAxpyF(length / 3, axpyFPref - 1);

    [TestMethod]
    public void TestContinueAxpyFEqual()
        => TestContinueAxpyF(length, axpyFPref);
    [TestMethod]
    public void TestContinueAxpyFMore()
        => TestContinueAxpyF(length, axpyFPref * 3 + 1);

    public static void TestContinueAxpyF(int m, int b)
    {
        double alpha = 2.0;
        var a = CreateMatrixRandom(m, b);
        var x = CreateVectorRandom(b);
        var y = CreateVectorRandom(m);
        var expected = CopyVector(y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < b; j++)
            {
                expected[i] += alpha * a[i, j] * x[j];
            }
        }

        BlasProvider.AxpyF(alpha, a, x, y);

        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    public static void TestSimpleAxpyF(int m, int b)
    {
        int stride = 2;
        double alpha = 2.0;
        var a = CreateMatrixStrideRandom(m, b, stride);
        var x = CreateVectorStrideRandom(b, stride);
        var y = CreateVectorRandom(m);
        var expected = CopyVector(y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < b; j++)
            {
                expected[i] += alpha * a[i, j] * x[j];
            }
        }

        BlasProvider.AxpyF(alpha, a, x, y);

        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleAxpyFLess()
        => TestSimpleAxpyF(length / 3, axpyFPref - 1);

    [TestMethod]
    public void TestSimpleAxpyFEqual()
        => TestSimpleAxpyF(length, axpyFPref);
    [TestMethod]
    public void TestSimpleAxpyFMore()
        => TestSimpleAxpyF(length, axpyFPref * 3 + 1);
    public static void TestTransAxpyF(int m, int b)
    {
        int stride = 2;
        double alpha = 2.0;
        var a = CreateMatrixTransRandom(m, b);
        var x = CreateVectorStrideRandom(b, stride);
        var y = CreateVectorRandom(m);
        var expected = CopyVector(y);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < b; j++)
            {
                expected[i] += alpha * a[i, j] * x[j];
            }
        }

        BlasProvider.AxpyF(alpha, a, x, y);

        for (int i = 0; i < m; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestTransAxpyFLess()
        => TestTransAxpyF(length / 3, axpyFPref - 1);

    [TestMethod]
    public void TestTransAxpyFEqual()
        => TestTransAxpyF(length, axpyFPref);
    [TestMethod]
    public void TestTransAxpyFMore()
        => TestTransAxpyF(length, axpyFPref * 3 + 1);
}
