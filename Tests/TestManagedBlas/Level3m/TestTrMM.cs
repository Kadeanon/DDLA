using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.Level3m;

[TestClass]
public class TestTrMM
{
    internal static readonly int length = Random.Shared.Next(513, 2048);

    [TestMethod]
    public void TestRowMajorUpperLeftTrMMLeft()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left, 
            UpLo.Upper);
    }

    [TestMethod]
    public void TestColMajorUpperTrMMLeft()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left, 
            UpLo.Upper);
    }

    [TestMethod]
    public void TestSimpleUpperTrMMLeft()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Left, 
            UpLo.Upper);
    }

    [TestMethod]
    public void TestRowMajorUpperTrMMLeftDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, side: SideType.Left,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorUpperTrMMLeftDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, side: SideType.Left,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMMLeft()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestColMajorLowerTrMMLeft()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestSimpleLowerTrMMLeft()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMMLeftDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorLowerTrMMLeftDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Left,
            UpLo.Lower, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorUpperTrMMRight()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper);
    }

    [TestMethod]
    public void TestColMajorUpperTrMMRight()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper);
    }

    [TestMethod]
    public void TestSimpleUpperTrMMRight()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper);
    }

    [TestMethod]
    public void TestRowMajorUpperTrMMRightDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorUpperTrMMRightDiag()
    {
        var mat = CreateMatrixTransRandom
            (length, length); 
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Upper, DiagType.Unit);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMMRight()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestColMajorLowerTrMMRight()
    {
        var mat = CreateMatrixTransRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestSimpleLowerTrMMRight()
    {
        int colStride = 4;
        var mat = CreateMatrixStrideRandom
            (length, length, colStride);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower);
    }

    [TestMethod]
    public void TestRowMajorLowerTrMMRightDiag()
    {
        var mat = CreateMatrixRandom
            (length, length);
        CheckExpectedResults(mat, SideType.Right,
            UpLo.Lower, DiagType.Unit);
    }

    [TestMethod]
    public void TestColMajorLowerTrMMRightDiag()
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
        double alpha = 1.2;
        var y = CreateMatrixRandom(length, length);
        var y2 = CopyMatrix(y);
        BlasProvider.TrMM(side, uplo, diag, alpha, a, y);
        BlisProvider.TrMM(side, uplo, diag, alpha, a, y2);
        var diff = y - y2;
        var maxValue = diff.Max(); 
        Assert.AreEqual(0, maxValue, 1e-10);
    }
}
