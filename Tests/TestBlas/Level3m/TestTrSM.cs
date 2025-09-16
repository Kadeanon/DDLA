
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestTrSM
{
    internal static int length = Random.Shared.Next(
        //2, 3);
        256, 512);

    [TestMethod]
    public void TestRowMajorUpperLeftTrSMLeft()
    {
        var mat = Matrix
            .RandomTriangle(length, UpLo.Upper);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestColMajorUpperTrSMLeft()
    {
        var mat =
            CreateMatrixTransRandom(length, length);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestSimpleUpperTrSMLeft()
    {
        int colStride = 4;
        var mat =
            CreateMatrixStrideRandom(length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat);
    }

    [TestMethod]
    public void TestRowMajorLowerTrSMLeft()
    {
        var mat = Matrix
            .RandomTriangle(length, UpLo.Lower);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestColMajorLowerTrSMLeft()
    {
        var mat =
            CreateMatrixTransRandom(length, length);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestSimpleLowerTrSMLeft()
    {
        int colStride = 4;
        var mat =
            CreateMatrixStrideRandom(length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat);
    }

    [TestMethod]
    public void TestRowMajorUpperTrSMRight()
    {
        var mat = Matrix.RandomTriangle(length, UpLo.Upper);
        CheckExpectedResults(UpLo.Upper, mat, side: SideType.Right);
    }

    [TestMethod]
    public void TestColMajorUpperTrSMRight()
    {
        var mat = CreateMatrixTransRandom(length, length);
        CheckExpectedResults(UpLo.Upper, mat, side: SideType.Right);
    }

    [TestMethod]
    public void TestSimpleUpperTrSMRight()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom(length, length, colStride);
        CheckExpectedResults(UpLo.Upper, mat, side: SideType.Right);
    }

    [TestMethod]
    public void TestRowMajorLowerTrSMRight()
    {
        var mat = Matrix.RandomTriangle(length, UpLo.Lower);
        CheckExpectedResults(UpLo.Lower, mat, side: SideType.Right);
    }

    [TestMethod]
    public void TestColMajorLowerTrSMRight()
    {
        var mat = CreateMatrixTransRandom(length, length);
        CheckExpectedResults(UpLo.Lower, mat, side: SideType.Right);
    }

    [TestMethod]
    public void TestSimpleLowerTrSMRight()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom(length, length, colStride);
        CheckExpectedResults(UpLo.Lower, mat, side: SideType.Right);
    }

    public static void CheckExpectedResults
        (UpLo uplo, MatrixView a, SideType side = SideType.Left)
    {
        int length = a.Rows;
        VectorView vec = a.Diag;
        var tol = 1e-8;
        for (int i = 0; i < length; i++)
        {
            vec[i] += 1;
        }
        double alpha = 1.0;
        var x = CreateMatrixRandom(length, length);
        var x2 = x.Clone();
        BlasProvider.TrMM(side, uplo, alpha, a, x2);
        BlasProvider.TrSM(side, uplo, alpha, a, x2);
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                var exp = x[i, j];
                var act = x2[i, j];
                var diff0 = Math.Abs(exp - act);
                Assert.AreEqual(0, diff0, tol, $"Mismatch at ({i}, {j}): expected {exp}, got {act}");
            }
        }
    }

}
