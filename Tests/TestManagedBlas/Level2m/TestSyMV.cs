using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

[TestClass]
public class TestSymv
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpSymv()
    {
        var mat = CreateMatrixRandom(length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpSymv()
    {
        var mat = CreateMatrixTransRandom(length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpSymv()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom(length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLoSymv()
    {
        var mat = CreateMatrixRandom(length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLoSymv()
    {
        var mat = CreateMatrixTransRandom(length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestSimpleLoSymv()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom(length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    public static void CheckExpectedResults(UpLo uplo, MatrixView a)
    {
        double alpha = 2.0;
        double beta = 1.2;
        int length = a.Cols;
        var x = CreateVectorRandom(length);
        var y = CreateVectorRandom(length);
        bool upper = uplo == UpLo.Upper;
        for (int i = 0; i < a.Rows; i++)
        {
            foreach (var j in upper ?
                Enumerable.Range(0, i) :
                Enumerable.Range(i + 1, length - i - 1))
            {
                a[i, j] = a[j, i];
            }
        }
        var result = CopyVector(y);
        BlasProvider.GeMV(alpha, a, x, beta, result);
        BlasProvider.SyMV(uplo, alpha, a, x, beta, y);
        for (int i = 0; i < a.Rows; i++)
        {
            Assert.AreEqual(result[i], y[i], 1e-10, $"Row {i} mismatch");
        }
    }

}
