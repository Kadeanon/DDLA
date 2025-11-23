using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level1v;

[TestClass]
public class TestAMaxV
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestContinueAMaxV_Common()
    {
        var x = CreateVectorRandom(length);
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestSimpleAMaxV_Common()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestContinueAMaxV_NaN_ShouldChooseNaN()
    {
        var x = CreateVectorRandom(length);
        x[Random.Shared.Next(length)] = double.NaN;
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestSimpleAMaxV_NaN_ShouldChooseNaN()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        x[Random.Shared.Next(length)] = double.NaN;
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestContinueAMaxV_TwoMax_ChooseFirst()
    {
        var x = CreateVectorRandom(length);

        var maxIndex = BlisProvider.AMax(x);
        var maxValue = x[maxIndex];
        if (maxIndex == length - 1)
            maxIndex--;
        else
            maxIndex++;
        x[maxIndex] = maxValue;
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestSimpleAMaxV_TwoMax_ChooseFirst()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);

        var maxIndex = BlisProvider.AMax(x);
        var maxValue = x[maxIndex];
        if (maxIndex == length - 1)
            maxIndex--;
        else
            maxIndex++;
        x[maxIndex] = maxValue;
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestContinueAMaxV_TwoNaN_ShouldChooseFirstNaN()
    {
        var x = CreateVectorRandom(length);
        int index0 = Random.Shared.Next(length);
        x[index0] = double.NaN;
        int index1 = Random.Shared.Next(length);
        while (index0 == index1) index1 = index0;
        x[index1] = double.NaN;
        CheckExpectedResult(x);
    }

    [TestMethod]
    public void TestSimpleAMaxV_TwoNaN_ShouldChooseFirstNaN()
    {
        int stride = 2;
        var x = CreateVectorStrideRandom(length, stride);
        int index0 = Random.Shared.Next(length);
        x[index0] = double.NaN;
        int index1 = Random.Shared.Next(length);
        while (index0 == index1) index1 = index0;
        x[index1] = double.NaN;
        CheckExpectedResult(x);
    }


    private static void CheckExpectedResult(VectorView x)
    {
        var expectedIndex = BlisProvider.AMax(x);
        var actualIndex = BlasProvider.AMax(x);

        Assert.AreEqual(expectedIndex, actualIndex, $"Expected index {expectedIndex}, but got {actualIndex}");
    }
}
