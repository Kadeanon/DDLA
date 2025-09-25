namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestTrMM
{
    internal static readonly int length = Random.Shared.Next(256, 1024);

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
        double alpha = 1.0;
        var x = CreateMatrixRandom
            (length, length);
        var y = x.Clone();
        var y2 = y.EmptyLike();
        for (int i = 0; i < length; i++)
        {
            a[i, i] = (0.5 + length) * (a[i, i] + 1);
        }
        BlasProvider.TrMM(side, uplo, 
            TransType.NoTrans, diag, alpha, a, y);
        BlasProvider.MakeTr(a, uplo);
        if (diag == DiagType.Unit) a.Diag.Fill(1);
        if(side == SideType.Left)
        {
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    var aik = a.GetRow(i);
                    var xkj = x.GetColumn(j);
                    var val = alpha * aik * xkj;
                    y2[i, j] = val;
                }
            }
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    var xik = x.GetRow(i);
                    var akj = a.GetColumn(j);
                    var val = alpha * xik * akj;
                    y2[i, j] = val;
                }
            }
        }
        var diff = y - y2;
        var maxValue = diff.Max(); 
        Assert.AreEqual(0, maxValue, 1e-10);
    }
}
