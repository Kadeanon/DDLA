using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1f;

[TestClass]
public class TestDotxAxpyF
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);
    internal static int dotxAxpyFPref = 8;

    public static void TestContinueDotxAxpyF(int m, int b)
    {
        double alpha = 2.0;
        double beta = Math.PI;
        var a = CreateMatrixRandom(m, b);
        var w = CreateVectorRandom(m);
        var x = CreateVectorRandom(b);
        var y = CreateVectorRandom(b);
        var z = CreateVectorRandom(m);
        var expectedY = CopyVector(y);
        var expectedZ = CopyVector(z);

        BlasProvider.Scal(beta, expectedY);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < m; i++)
            {
                expectedY[j] += alpha * a[i, j] * w[i];
                expectedZ[i] += alpha * a[i, j] * x[j];
            }
        }

        BlasProvider.DotxAxpyF(alpha, a, w, x, beta, y, z);

        for (int i = 0; i < b; i++)
            Assert.AreEqual(expectedY[i], y[i], Tolerance, $"Index {i} mismatch");
        for (int i = 0; i < m; i++)
            Assert.AreEqual(expectedZ[i], z[i], Tolerance, $"Index {i} mismatch");
    }

    [TestMethod]
    public void TestContinueDotxAxpyFLess()
        => TestContinueDotxAxpyF(length / 3, dotxAxpyFPref - 1);

    [TestMethod]
    public void TestContinueDotxAxpyFEqual()
        => TestContinueDotxAxpyF(length, dotxAxpyFPref);

    [TestMethod]
    public void TestContinueDotxAxpyFMore()
        => TestContinueDotxAxpyF(length, dotxAxpyFPref * 3 + 1);

    public static void TestSimpleDotxAxpyF(int m, int b)
    {
        double alpha = 2.0;
        int stride = 3;
        double beta = Math.PI;
        var a = CreateMatrixStrideRandom(m, b, stride);
        var w = CreateVectorRandom(m);
        var x = CreateVectorRandom(b);
        var y = CreateVectorRandom(b);
        var z = CreateVectorRandom(m);
        var expectedY = CopyVector(y);
        var expectedZ = CopyVector(z);

        BlasProvider.Scal(beta, expectedY);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < m; i++)
            {
                expectedY[j] += alpha * a[i, j] * w[i];
                expectedZ[i] += alpha * a[i, j] * x[j];
            }
        }

        BlasProvider.DotxAxpyF(alpha, a, w, x, beta, y, z);

        for (int i = 0; i < b; i++)
            Assert.AreEqual(expectedY[i], y[i], Tolerance, $"Index {i} mismatch");
        for (int i = 0; i < m; i++)
            Assert.AreEqual(expectedZ[i], z[i], Tolerance, $"Index {i} mismatch");
    }

    [TestMethod]
    public void TestSimpleDotxAxpyFLess()
        => TestSimpleDotxAxpyF(length / 3, dotxAxpyFPref - 1);

    [TestMethod]
    public void TestSimpleDotxAxpyFEqual()
        => TestSimpleDotxAxpyF(length, dotxAxpyFPref);

    [TestMethod]
    public void TestSimpleDotxAxpyFMore()
        => TestSimpleDotxAxpyF(length, dotxAxpyFPref * 3 + 1);

    public static void TestTransDotxAxpyF(int m, int b)
    {
        double alpha = 2.0;
        double beta = Math.PI;
        var a = CreateMatrixTransRandom(m, b);
        var w = CreateVectorRandom(m);
        var x = CreateVectorRandom(b);
        var y = CreateVectorRandom(b);
        var z = CreateVectorRandom(m);
        var expectedY = CopyVector(y);
        var expectedZ = CopyVector(z);

        BlasProvider.Scal(beta, expectedY);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < m; i++)
            {
                var aVal = alpha * a[i, j];
                expectedY[j] += aVal * w[i];
                expectedZ[i] += aVal * x[j];
            }
        }

        BlasProvider.DotxAxpyF(alpha, a, w, x, beta, y, z);

        for (int i = 0; i < b; i++)
            Assert.AreEqual(expectedY[i], y[i], Tolerance, $"Index {i} mismatch");
        for (int i = 0; i < m; i++)
            Assert.AreEqual(expectedZ[i], z[i], Tolerance, $"Index {i} mismatch");
    }

    [TestMethod]
    public void TestTransDotxAxpyFLess()
        => TestTransDotxAxpyF(length / 3, dotxAxpyFPref - 1);

    [TestMethod]
    public void TestTransDotxAxpyFEqual()
        => TestTransDotxAxpyF(length, dotxAxpyFPref);

    [TestMethod]
    public void TestTransDotxAxpyFMore()
        => TestTransDotxAxpyF(length, dotxAxpyFPref * 3 + 1);
}
