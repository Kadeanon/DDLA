using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1f;

[TestClass]
public class TestDotxF
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);
    internal static int dotxFPref = 8;

    public static void TestContinueDotxF(int m, int b)
    {
        double alpha = 2.0;
        double beta = Math.PI;
        var a = CreateMatrixRandom(m, b);
        var x = CreateVectorRandom(m);
        var y = CreateVectorRandom(b);
        var expected = CopyVector(y);

        BlasProvider.Scal(beta, expected);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < m; i++)
            {
                expected[j] += alpha * a[i, j] * x[i];
            }
        }

        BlasProvider.DotxF(alpha, a, x, beta, y);

        for (int i = 0; i < b; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestContinueDotxFLess()
        => TestContinueDotxF(length / 3, dotxFPref - 1);

    [TestMethod]
    public void TestContinueDotxFEqual()
        => TestContinueDotxF(length, dotxFPref);

    [TestMethod]
    public void TestContinueDotxFMore()
        => TestContinueDotxF(length, dotxFPref * 3 + 1);

    public static void TestSimpleDotxF(int m, int b)
    {
        int stride = 2;
        double alpha = 2.0;
        double beta = Math.PI;
        var a = CreateMatrixStrideRandom(m, b, stride);
        var x = CreateVectorStrideRandom(m, stride);
        var y = CreateVectorRandom(b);
        var expected = CopyVector(y);

        BlasProvider.Scal(beta, expected);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < m; i++)
            {
                expected[j] += alpha * a[i, j] * x[i];
            }
        }

        BlasProvider.DotxF(alpha, a, x, beta, y);

        for (int i = 0; i < b; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleDotxFLess()
        => TestSimpleDotxF(length / 3, dotxFPref - 1);

    [TestMethod]
    public void TestSimpleDotxFEqual()
        => TestSimpleDotxF(length, dotxFPref);

    [TestMethod]
    public void TestSimpleDotxFMore()
        => TestSimpleDotxF(length, dotxFPref * 3 + 1);

    public static void TestTransDotxF(int m, int b)
    {
        int stride = 2;
        double alpha = 2.0;
        double beta = 0;// Math.PI;
        var a = CreateMatrixTransRandom(m, b);
        var x = CreateVectorStrideRandom(m, stride);
        var y = CreateVectorRandom(b);
        var expected = CopyVector(y);

        for (int j = 0; j < b; j++)
        {
            double temp = 0.0;
            for (int i = 0; i < m; i++)
            {
                temp += a[i, j] * x[i];
            }
            expected[j] *= beta;
            expected[j] += temp * alpha;
        }

        BlasProvider.DotxF(alpha, a, x, beta, y);

        for (int i = 0; i < b; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestTransDotxFLess()
        => TestTransDotxF(length / 3, dotxFPref - 1);

    [TestMethod]
    public void TestTransDotxFEqual()
        => TestTransDotxF(length, dotxFPref);

    [TestMethod]
    public void TestTransDotxFMore()
        => TestTransDotxF(length, dotxFPref * 3 + 1);
}
