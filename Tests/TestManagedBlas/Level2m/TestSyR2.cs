using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace Tests.TestManagedBlas.Level2m;

[TestClass]
public class TestSyr2
{
    internal static int length = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestRowMajorUpSyR2()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpSyR2()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpSyR2()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLoSyR2()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLoSyR2()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [Ignore("Maybe aocl bug")]
    [TestMethod]
    public void TestSimpleLoSyR2()
    {
        int colStride = 2;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    public static void CheckExpectedResults(UpLo uplo, MatrixView a)
    {
        double alpha = 2.0;
        int cols = a.Cols, rows = a.Rows;
        var vecx = CreateVectorRandom(rows);
        var vecy = CreateVectorRandom(cols);
        bool upper = uplo == UpLo.Upper;
        var result = CopyMatrix(a);
        BlasProvider.SyR2(uplo, alpha, vecx, vecy, a);
        BlasProvider.GeR(alpha, vecx, vecy, result);
        BlasProvider.GeR(alpha, vecy, vecx, result);
        if (upper)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = i; j < a.Cols; j++)
                {
                    Assert.AreEqual(result[i, j], a[i, j], 1e-10, $"Matrix {i},{j} mismatch");
                }
            }
        }
        else
        {
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    Assert.AreEqual(result[i, j], a[i, j], 1e-10, $"Matrix {i},{j} mismatch");
                }
            }
        }
    }
}
