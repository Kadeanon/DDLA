using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1v;

[TestClass]
public class TestXpbyV
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueXpbyV()
    {
        double beta = 1.5;
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        var expected = CopyVector(y);

        for (int i = 0; i < length; i++)
        {
            expected[i] = x[i] + beta * y[i];
        }

        BlasProvider.Xpby(x, beta, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleXpbyV()
    {
        int stride = 2;
        double beta = 1.5;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        var expected = CopyVector(y);

        for (int i = 0; i < length; i++)
        {
            expected[i] = x[i] + beta * y[i];
        }

        BlasProvider.Xpby(x, beta, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
