using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level3m;

[TestClass]
public class TestGeMMT
{
    internal static int size = Random.Shared.Next(513, 2048);
    internal static int reds = Random.Shared.Next(513, 2048);

    [TestMethod]
    public void TestRowMajorUpperGeMMT()
    {
        var left =
            CreateMatrixRandom(size, reds);
        var right =
            CreateMatrixRandom(reds, size);
        var result =
            CreateMatrix(size, size);
        CheckExpectedResults
            (UpLo.Upper, left, right, result);
    }

    [TestMethod]
    public void TestColMajorUpperGeMMT()
    {
        var left =
            CreateMatrixTransRandom(size, reds);
        var right =
            CreateMatrixTransRandom(reds, size);
        var result =
            CreateMatrix(size, size);
        CheckExpectedResults
            (UpLo.Upper, left, right, result);
    }

    [TestMethod]
    public void TestSimpleUpperGeMMT()
    {
        int colStride = 4;
        var size = 4;
        var reds = 4;
        var left =
            CreateMatrixStrideRandom(size, reds, colStride);
        var right =
            CreateMatrixTransRandom(reds, size);
        var result =
            CreateMatrix(size, size, colStride);
        CheckExpectedResults
            (UpLo.Upper, left, right, result);
    }

    [TestMethod]
    public void TestRowMajorLowerGeMMT()
    {
        var left =
            CreateMatrixRandom(size, reds);
        var right =
            CreateMatrixRandom(reds, size);
        var result =
            CreateMatrix(size, size);
        CheckExpectedResults
            (UpLo.Lower, left, right, result);
    }

    [TestMethod]
    public void TestColMajorLowerGeMMT()
    {
        var left =
            CreateMatrixTransRandom(size, reds);
        var right =
            CreateMatrixTransRandom(reds, size);
        var result =
            CreateMatrix(size, size);
        CheckExpectedResults
            (UpLo.Lower, left, right, result);
    }

    [TestMethod]
    public void TestSimpleLowerGeMMT()
    {
        int colStride = 4;
        var size = 4;
        var reds = 4;
        var left =
            CreateMatrixStrideRandom(size, reds, colStride);
        var right =
            CreateMatrixStrideRandom(reds, size, colStride);
        var result =
            CreateMatrix(size, size, colStride);
        left.Fill(1);
        right.Fill(1);
        CheckExpectedResults
            (UpLo.Lower, left, right, result);
    }

    public static void CheckExpectedResults
        (UpLo uplo, MatrixView left, MatrixView right, 
        MatrixView result)
    {
        double alpha = 2.0;
        double beta = 1.2;
        var expected = CopyMatrix(result);
        int size = left.Rows;
        BlisProvider.GeMMT(uplo, alpha, left, right, beta, expected);
        BlasProvider.GeMMT(uplo, alpha, left, right, beta, result);
        for (int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
            {
                Assert.AreEqual(expected[i, j], result[i, j], 
                    1e-10, $"Value ({i}, {j}) mismatch");
            }
        }
    }

}
