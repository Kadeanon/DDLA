using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;

namespace Tests.TestBlas.Level3m;

[TestClass]
public class TestSyMM
{
    internal static int rows = Random.Shared.Next(256, 1024);
    internal static int cols = Random.Shared.Next(256, 1024);

    [TestMethod]
    public void TestUpperSyMMLeft()
    {
        CheckExpectedResults
            (UpLo.Upper, SideType.Left);
    }

    [TestMethod]
    public void TestLowerSyMMLeft()
    {
        CheckExpectedResults
            (UpLo.Lower, SideType.Left);
    }

    [TestMethod]
    public void TestUpperSyMMRight()
    {
        CheckExpectedResults
            (UpLo.Upper, SideType.Right);
    }

    [TestMethod]
    public void TestLowerSyMMRight()
    {
        CheckExpectedResults(UpLo.Lower, SideType.Right);
    }

    public static void CheckExpectedResults(UpLo uplo, SideType side)
    {
        int large = 768;
        int small = 64;
        CheckExpectedResults
            (uplo, side, large, large);
        CheckExpectedResults
            (uplo, side, large, small);
        CheckExpectedResults
            (uplo, side, small, large);
        CheckExpectedResults
            (uplo, side, small, small);
    }
    public static void CheckExpectedResults(UpLo uplo, SideType side, int m, int n)
    {
        double alpha = 1 / Math.PI;
        double beta = Math.E;
        int k = side == SideType.Left ? m : n;
        var A = CreateMatrixRandom(k, k);
        BlasProvider.MakeSy(A, uplo);
        var B = CreateMatrixRandom(m, n);
        var C = CreateMatrixRandom(m, n);
        var CExpected = C.Clone();
        BlasProvider.SyMM(side, uplo, alpha, A, B, beta, C);
        if (side == SideType.Left)
            BlasProvider.GeMM(alpha, A, B, beta, CExpected);
        else
            BlasProvider.GeMM(alpha, B, A, beta, CExpected);
        var diff = CExpected - C;
        var norm = diff.View.Nrm1();
        Assert.AreEqual(0, norm, 2e-5, $"C mismatch in {m}x{n} state.");
    }
}
