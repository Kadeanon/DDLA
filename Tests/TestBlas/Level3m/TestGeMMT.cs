using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestGeMMT
{
    internal static int size = Random.Shared.Next(64, 1024);
    internal static int reds = Random.Shared.Next(64, 1024);
    
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

    [Ignore("Maybe aocl bug")]
    [TestMethod]
    public void TestSimpleUpperGeMMT()
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

    [Ignore("Maybe aocl bug")]
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
        BlasProvider.GeMM(alpha, left, right, beta, expected);
        BlasProvider.GeMMt(uplo, alpha, left, right, beta, result);
        BlasProvider.Sub(DiagType.NonUnit,
            uplo, TransType.NoTrans, expected, result);
        for (int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
            {
                Assert.AreEqual(0, result[i, j], 1e-10, $"Value ({i}, {j}) mismatch");
            }
        }
    }

}
