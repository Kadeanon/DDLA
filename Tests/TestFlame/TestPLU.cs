
using DDLA.Factorizations;

namespace Tests.TestFlame;

public partial class TestLU
{
    #region partpiv
    [TestMethod]
    public void TestMediumBloPLU()
        => TestPartPivLU(medium, LU.PLUDecBlock);
    
    [TestMethod]
    public void TestSmallUnbPLU()
        => TestPartPivLU(small, LU.PLUDecUnblock);
    
    [TestMethod]
    public void TestTinyUnbPLU()
        => TestPartPivLU(tiny, LU.PLUDecUnblock);

    [TestMethod]
    public void TestLargePLU()
    {
        Random random = new(seed);
        var count = 5;
        double tol = 1e-6;
        for (int i = 0; i < count; i++)
        {
            var A = Matrix.RandomSPD(large, random).View;
            var plu = new LU(A);
            var (P, L, U) = plu;
            var diff = L * U - P * A;
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
    public void TestLargePLUSolver()
    {
        Random random = new(seed);
        var count = 5;
        double tol = 1e-6;
        for (int i = 0; i < count; i++)
        {
            var A = Matrix.RandomDense(large, large, random).View;
            var X = Matrix.Eyes(large).View;
            var B = A * X;
            var plu = new LU(A);
            var X2 = plu.Solve(B);
            var diff = X - X2;
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

    #endregion partpiv

    #region partpiv static
    
    private static void TestPartPivLU(int length, 
        Action<MatrixView, Span<int>> action, 
        double tol = 1e-8)
    {
        Random random = new(seed);
        var count = 20;
        for (int i = 0; i < count; i++)
        {
            var lower = CreateMatrixRandom
            (length, length, random).View;
            var upper = CreateMatrixRandom
            (length, length, random).View;
            MakeLMat(lower);
            MakeUMat(upper);
            var orig = lower * upper;
            orig.CopyTo(lower);
            var pivs = new int[length];
            action(lower, pivs);
            lower.CopyTo(upper);
            MakeLMat(lower);
            MakeUMat(upper);
            var result = lower * upper;
            var p = Matrix.Eyes(length);
            LU.ApplyPiv(p, pivs);
            var diff = p * orig - result;
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
    #endregion partpiv static

}
