using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1f;

[TestClass]
public class TestAxpy2
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueAxpy2()
    {
        double alphaX = 2.0;
        double alphaY = Math.PI;
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        var z = CreateVectorRandom(length);
        var expected = CopyVector(z);

        for (int i = 0; i < length; i++)
        {
            expected[i] += alphaX * x[i] + alphaY * y[i];
        }

        BlasProvider.Axpy2(alphaX, alphaY, x, y, z);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], z[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleAxpy2()
    {
        int stride = 2;
        double alphaX = 2.0;
        double alphaY = Math.PI;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        var z = CreateVectorRandom(length);
        var expected = CopyVector(z);

        for (int i = 0; i < length; i++)
        {
            expected[i] += alphaX * x[i] + alphaY * y[i];
        }

        BlasProvider.Axpy2(alphaX, alphaY, x, y, z);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], z[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
