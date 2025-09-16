namespace Tests.TestBlas.Level1v;

[TestClass]
public class TestCopyV
{
    private const double Tolerance = double.Epsilon;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueCopyV()
    {
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);

        BlasProvider.Copy(x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(x[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }

    [TestMethod]
    public void TestSimpleCopyV()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);

        BlasProvider.Copy(x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(x[i], y[i], Tolerance, $"Index {i} mismatch");
        }
    }
}
