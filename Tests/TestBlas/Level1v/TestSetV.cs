namespace Tests.TestBlas.Level1v;

[TestClass]
public class TestSetV
{
    private const double Tolerance = double.Epsilon;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueSetV()
    {
        double alpha = 2.0;
        var x = CreateVectorRandom(length);

        BlasProvider.Set(alpha, x);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(alpha, x[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleSetV()
    {
        int stride = 2;
        double alpha = 2.0;
        var x = CreateVectorStrideRandom(length, stride);

        BlasProvider.Set(alpha, x);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(alpha, x[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
