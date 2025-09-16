namespace Tests.TestBlas.Level1v;

[TestClass]
public class TestInvertV
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueInvertV()
    {
        var x = CreateVectorRandom(length);
        var expected = CopyVector(x);

        for (int i = 0; i < length; i++)
        {
            expected[i] = 1.0 / x[i];
        }

        BlasProvider.Invert(x);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], x[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleInvertV()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        var expected = CopyVector(x);

        for (int i = 0; i < length; i++)
        {
            expected[i] = 1.0 / x[i];
        }

        BlasProvider.Invert(x);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], x[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
