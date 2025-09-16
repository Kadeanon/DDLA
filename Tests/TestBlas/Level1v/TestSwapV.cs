namespace Tests.TestBlas.Level1v;

[TestClass]
public class TestSwapV
{
    private const double Tolerance = double.Epsilon;
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueSwapV()
    {
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        var expectedX = CopyVector(y);
        var expectedY = CopyVector(x);

        BlasProvider.Swap(x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expectedX[i], x[i], Tolerance, $"Index {i} mismatch in x");
            Assert.AreEqual(expectedY[i], y[i], Tolerance, $"Index {i} mismatch in y");
        }
    }

    [TestMethod]
    public void TestSimpleSwapV()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        var y = CreateVectorStrideRandom(length, stride);
        var expectedX = CopyVector(y);
        var expectedY = CopyVector(x);

        BlasProvider.Swap(x, y);

        for (int i = 0; i < length; i++)
        {
            Assert.AreEqual(expectedX[i], x[i], Tolerance, $"Index {i} mismatch in x");
            Assert.AreEqual(expectedY[i], y[i], Tolerance, $"Index {i} mismatch in y");
        }
    }
}
