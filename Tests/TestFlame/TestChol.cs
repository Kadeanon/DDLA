using DDLA.Factorizations;

namespace Tests.TestFlame;

[TestClass]
public class TestChol
{
    private const int tiny = 4;
    private const int small = 64;
    private const int medium = 256;
    private const int large = 1024;

    #region Lower Cholesky Decomposition Tests
    [TestMethod]
    public void TestTinyUnbCholLower()
        => TestCholLowerDec(tiny, Cholesky.CholeskyLowerUnblock);

    [TestMethod]
    public void TestSmallUnbCholLower()
        => TestCholLowerDec(small, Cholesky.CholeskyLowerUnblock);

    [TestMethod]
    public void TestMediumBloCholLower()
        => TestCholLowerDec(medium, Cholesky.CholeskyLowerBlock);

    private static void TestCholLowerDec(int m, Action<MatrixView> action, double tol = 1e-10)
    {
        int count = 20;
        for (int i = 0; i < count; i++)
        {
            var L =
                CreateMatrixRandom(m, m).View;
            //L.MakeTr(UpLo.Lower);
            CheckForNaN(L);
            var orig = L * L.T;
            orig.CopyTo(L);
            CheckForNaN(L);
            action(L);
            CheckForNaN(L);
            BlasProvider.MakeTr(L, UpLo.Lower);
            var result = L * L.T;
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
    #endregion Lower Cholesky Decomposition Tests

    #region Upper Cholesky Decomposition Tests
    [TestMethod]
    public void TestTinyUnbCholUpper()
        => TestCholUpperDec(tiny, Cholesky.CholeskyUpperUnblock);

    [TestMethod]
    public void TestSmallUnbCholUpper()
        => TestCholUpperDec(small, Cholesky.CholeskyUpperUnblock);

    [TestMethod]
    public void TestMediumBloCholUpper()
        => TestCholUpperDec(medium, Cholesky.CholeskyUpperBlock);

    private static void TestCholUpperDec(int m, Action<MatrixView> action, double tol = 1e-10)
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
            var result = U.T * U;
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
    #endregion Upper Cholesky Decomposition Tests


    #region Cholesky Solver Tests
    [TestMethod]
    public void TestLargeCholLowerSolver()
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
            var XSolve = new Cholesky(A, UpLo.Lower).Solve(B);
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
    public void TestLargeCholUpperSolver()
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
            var XSolve = new Cholesky(A, UpLo.Upper).Solve(B);
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
    #endregion Cholesky Solver Tests

    private static void CheckForNaN(MatrixView A)
    {
        for (int j = 0; j < A.Cols; j++)
        {
            for (int i = 0; i < A.Rows; i++)
            {
                Assert.IsFalse(double.IsNaN(A[i, j]),
                    $"Matrix contains NaN at ({i}, {j}).");
            }
        }
    }
}
