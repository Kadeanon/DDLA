namespace Tests.TestBlas.Level1v;

[TestClass]
public class TestScalV
{
    private const double Tolerance = 1e-10;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueScalV()
    {
        double alpha = 2.0;
        var x = CreateVectorRandom(length);
        var expected = CopyVector(x);

        for (int i = 0; i < length; i++)
        {
            expected[i] *= alpha;
        }

        BlasProvider.Scal(alpha, x);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], x[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleScalV()
    {
        int stride = 2;
        double alpha = 2.0;
        var x = CreateVectorStrideRandom(length, stride);
        var expected = CopyVector(x);

        for (int i = 0; i < length; i++)
        {
            expected[i] *= alpha;
        }

        BlasProvider.Scal(alpha, x);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expected[i], x[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
