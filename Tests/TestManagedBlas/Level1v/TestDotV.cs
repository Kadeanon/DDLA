using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1v;

[TestClass]
public class TestDotV
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueDotV()
    {
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        double expected = 0.0;

        for (int i = 0; i < length; i++)
        {
            expected += x[i] * y[i];
        }

        double result = BlasProvider.Dot(x, y);

        Assert.AreEqual(expected, result, Tolerance, "Dot product mismatch");
    }

    [TestMethod]
    public void TestSimpleDotV()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        double expected = 0.0;

        for (int i = 0; i < length; i++)
        {
            expected += x[i] * y[i];
        }

        double result = BlasProvider.Dot(x, y);

        Assert.AreEqual(expected, result, Tolerance, "Dot product mismatch");
    }
}
