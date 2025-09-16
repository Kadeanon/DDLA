namespace Tests.TestBlas.Level1v;

[TestClass]
public class TestAMaxV
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueAMaxV()
    {

        var x = CreateVectorRandom(length);
        var expectedIndex = 0;
        double maxValue = x[0];

        for (int i = 1; i < length; i++)
        {
            if (x[i] > maxValue)
            {
                maxValue = x[i];
                expectedIndex = i;
            }
        }

        var actualIndex = BlasProvider.AMax(x);

        Assert.AreEqual(expectedIndex, actualIndex, 
            $"Expected index {expectedIndex}, but got {actualIndex}");
    }

    [TestMethod]
    public void TestSimpleAMaxV()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        var expectedIndex = 0;
        double maxValue = x[0];

        for (int i = 1; i < length; i++)
        {
            if (x[i] > maxValue)
            {
                maxValue = x[i];
                expectedIndex = i;
            }
        }

        var actualIndex = BlasProvider.AMax(x);

        Assert.AreEqual(expectedIndex, actualIndex, $"Expected index {expectedIndex}, but got {actualIndex}");
    }
}
