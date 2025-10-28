using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level1v;

[TestClass]
public class TestSubV
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueSubV()
    {
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        var expected = CopyVector(y);

        for (int i = 0; i < length; i++)
        {
            expected[i] -= x[i];
        }

        BlasProvider.Sub(x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleSubV()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        var expected = CopyVector(y);

        for (int i = 0; i < length; i++)
        {
            expected[i] -= x[i];
        }

        BlasProvider.Sub(x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
