using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level3m;

[TestClass]
public class TestTrSM
{
    internal static int length = Random.Shared.Next(513, 768);

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
        var tol = 1e-8;
        BlisProvider.ShiftDiag(2, a);
        double alpha = 1.2;
        var result = CreateMatrixRandom(length, length);
        BlasProvider.TrMM(side, uplo, alpha, a, result);
        var expected = result.Clone();
        BlasProvider.TrSM(side, uplo, alpha, a, result);
        BlisProvider.TrSM(side, uplo, alpha, a, expected);
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                var exp = result[i, j];
                var act = expected[i, j];
                Assert.AreEqual(exp, act, tol, $"Mismatch at ({i}, {j}): " +
                    $"diff {exp - act}");
            }
        }
    }
}
