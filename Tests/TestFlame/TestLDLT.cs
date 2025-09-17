using DDLA.Factorizations;

namespace Tests.TestFlame;

[TestClass]
public class TestLDLT
{
    private const int tiny = 4;
    private const int small = 64;
    private const int medium = 256;
    private const int large = 1024;

    #region Lower LDLT Decomposition Tests
    [TestMethod]
    public void TestTinyUnbLDLTLower()
        => TestLDLTLowerDec(tiny, LDLT.LDLTLowerUnblock);

    [TestMethod]
    public void TestSmallUnbLDLTLower()
        => TestLDLTLowerDec(small, LDLT.LDLTLowerUnblock);

    //[TestMethod]
    //public void TestMediumBloLDLTLower()
    //    => TestLDLTLowerDec(medium, LDLT.LDLTLowerBlock);

    private static void TestLDLTLowerDec(int m, Action<MatrixView> action, double tol = 1e-10)
    {
        int count = 20;
        for (int i = 0; i < count; i++)
        {
            var L =
                CreateMatrixRandom(m, m).View;
            var orig = L * L.T;
            orig.CopyTo(L);
            action(L);
            BlasProvider.MakeTr(L, UpLo.Lower);
            var D = Matrix.Diagonals(L.Diag);
            L.Diag.Fill(1.0);
            var result = L * D * L.T;
            var diff = orig - result;
            double n1 = diff.Nrm1(),
                n2 = diff.NrmF(),
                nf = diff.NrmInf();
            Assert.IsTrue(n1 < tol,
                $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol,
                $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol,
                $"NormInf error {nf} exceeds tolerance {tol}.");
            Console.WriteLine($"Test {i + 1}/{count}:" +
                $"n1 = {n1}, n2 = {n2}, nf = {nf}");
        }
    }
    #endregion Lower LDLT Decomposition Tests

    #region Upper LDLT Decomposition Tests
    [TestMethod]
    public void TestTinyUnbLDLTUpper()
        => TestLDLTUpperDec(tiny, LDLT.LDLTUpperUnblock);

    [TestMethod]
    public void TestSmallUnbLDLTUpper()
        => TestLDLTUpperDec(small, LDLT.LDLTUpperUnblock);

    //[TestMethod]
    //public void TestMediumBloLDLTUpper()
    //    => TestLDLTUpperDec(medium, LDLT.LDLTUpperBlock);

    private static void TestLDLTUpperDec(int m, Action<MatrixView> action, double tol = 1e-10)
    {
        int count = 20;
        for (int i = 0; i < count; i++)
        {
            var U =
                CreateMatrixRandom(m, m).View;
            var orig = U.T * U;
            orig.CopyTo(U);
            action(U);
            BlasProvider.MakeTr(U, UpLo.Upper);
            var D = Matrix.Diagonals(U.Diag);
            U.Diag.Fill(1.0);
            var result = U.T * D * U;
            var diff = orig - result;
            double n1 = diff.Nrm1(),
                n2 = diff.NrmF(),
                nf = diff.NrmInf();
            Assert.IsTrue(n1 < tol,
                $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol,
                $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol,
                $"NormInf error {nf} exceeds tolerance {tol}.");
            Console.WriteLine($"Test {i + 1}/{count}:" +
                $"n1 = {n1}, n2 = {n2}, nf = {nf}");
        }
    }
    #endregion Upper LDLT Decomposition Tests


    #region LDLT Solver Tests
    [TestMethod]
    public void TestLargeLDLTLowerSolver()
    {
        int count = 5;
        double tol = 1e-8;
        for (int i = 0; i < count; i++)
        {
            var len = large + i;
            var U =
                CreateMatrixRandom(len, len).View;
            var A = U * U.T;
            A.Diag += CreateVector
                (len, Math.Sqrt(len));
            var X = CreateMatrixRandom
                (len, medium).View;
            var B = A * X;
            var XSolve = new LDLT(UpLo.Lower, A).Solve(B);
            var diff = XSolve - X;
            double n1 = diff.Nrm1(),
                n2 = diff.NrmF(),
                nf = diff.NrmInf();
            Assert.IsTrue(n1 < tol,
                $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol,
                $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol,
                $"NormInf error {nf} exceeds tolerance {tol}.");
            Console.WriteLine($"Test {i + 1}/{count}:" +
                $"n1 = {n1}, n2 = {n2}, nf = {nf}");
        }
    }

    [TestMethod]
    public void TestLargeLDLTUpperSolver()
    {
        int count = 5;
        double tol = 1e-8;
        for (int i = 0; i < count; i++)
        {
            var U =
                CreateMatrixRandom(large, large).View;
            var A = U.T * U;
            A.Diag += CreateVector
                (large, Math.Sqrt(large));
            var X =
                CreateMatrixRandom(large, medium).View;
            var B = A * X;
            var XSolve = new LDLT(UpLo.Upper, A).Solve(B);
            var diff = XSolve - X;
            double n1 = diff.Nrm1(),
                n2 = diff.NrmF(),
                nf = diff.NrmInf();
            Assert.IsTrue(n1 < tol,
                $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol,
                $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol,
                $"NormInf error {nf} exceeds tolerance {tol}.");
            Console.WriteLine($"Test {i + 1}/{count}:" +
                $"n1 = {n1}, n2 = {n2}, nf = {nf}");
        }
    }
    #endregion LDLT Solver Tests


}
