using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1f;

[TestClass]
public class TestDotAxpy
{
    private const double AxpyTolerance = 1e-10;
    private const double DotTolerance = 1e-8;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueDotAxpy()
    {
        double alpha = Math.PI;
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        var z = CreateVectorRandom(length);
        var expectedZ = CopyVector(z);
        var expectedRho = 0.0;

        for (int i = 0; i < length; i++)
        {
            expectedRho += x[i] * y[i];
            expectedZ[i] += alpha * x[i];
        }
        var rho = 0.0;
        BlasProvider.DotAxpy(alpha, x, y, ref rho, z);

        Assert.AreEqual(expectedRho, rho, DotTolerance, "Dot result mismatch");
        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expectedZ[i], z[i], AxpyTolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleDotAxpy()
    {
        int stride = 2;
        double alpha = Math.PI;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        var z = CreateVectorRandom(length);
        var expectedZ = CopyVector(z);
        var expectedRho = 0.0;

        for (int i = 0; i < length; i++)
        {
            expectedRho += x[i] * y[i];
            expectedZ[i] += alpha * x[i];
        }
        var rho = 0.0;
        BlasProvider.DotAxpy(alpha, x, y, ref rho, z);

        Assert.AreEqual(expectedRho, rho, DotTolerance, "Dot result mismatch");
        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expectedZ[i], z[i], AxpyTolerance, $"Index {i} mismatch");
        }
    }
}
