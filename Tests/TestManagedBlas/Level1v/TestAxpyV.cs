using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1v;

[TestClass]
public class TestAxpyV
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueAxpyV()
    {
        double alpha = 2.0;
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        var expected = CopyVector(y);

        for (int i = 0; i < length; i++)
        {
            expected[i] += alpha * x[i];
        }

        BlasProvider.Axpy(alpha, x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleAxpyV()
    {
        int stride = 2;
        double alpha = 2.0;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        var expected = CopyVector(y);

        for (int i = 0; i < length; i++)
        {
            expected[i] += alpha * x[i];
        }

        BlasProvider.Axpy(alpha, x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
