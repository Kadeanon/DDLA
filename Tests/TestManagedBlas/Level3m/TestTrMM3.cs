using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level3m;

[TestClass]
public class TestTrMM3
{
    internal static readonly int length = Random.Shared.Next(513, 2048);

    [TestMethod]
    public void TestRowMajorUpperLeftTrMM3Left()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left, 
            UpLo.Upper);
    }

    [TestMethod]
    public void TestColMajorUpperTrMM3Left()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left, 
            UpLo.Upper);
    }

    [TestMethod]
    public void TestSimpleUpperTrMM3Left()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Left, 
            UpLo.Upper);
    }

    [TestMethod]
    public void TestRowMajorUpperTrMM3LeftDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, side: SideType.Left,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorUpperTrMM3LeftDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, side: SideType.Left,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMM3Left()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestColMajorLowerTrMM3Left()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestSimpleLowerTrMM3Left()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMM3LeftDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorLowerTrMM3LeftDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorUpperTrMM3Right()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper);
    }

    [TestMethod]
    public void TestColMajorUpperTrMM3Right()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper);
    }

    [TestMethod]
    public void TestSimpleUpperTrMM3Right()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper);
    }

    [TestMethod]
    public void TestRowMajorUpperTrMM3RightDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorUpperTrMM3RightDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length); 
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMM3Right()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestColMajorLowerTrMM3Right()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestSimpleLowerTrMM3Right()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMM3RightDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorLowerTrMM3RightDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower, DiagType.Unit);
    }

    public static void CheckExpectedResults
        (MatrixView a, SideType side, UpLo uplo, 
        DiagType diag = DiagType.NonUnit)
    {
        double alpha = 1.0;
        double beta = 1.0;
        var x = CreateMatrixRandom(length, length);
        var y = CreateMatrixRandom(length, length);
        var y2 = y.Clone().View;
        BlisProvider.TrMM3(side, uplo, diag, alpha, a, x, beta, y2);
        BlasProvider.TrMM3(side, uplo, diag, alpha, a, x, beta, y);

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                Assert.AreEqual(y2[i, j], y[i, j], 1e-10,
                    $"Mismatch at ({i},{j})");
            }
        }
    }
}
