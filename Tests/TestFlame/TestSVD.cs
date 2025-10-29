using DDLA.Core;
using DDLA.Factorizations;

namespace Tests.TestFlame;

[TestClass]
public class TestSVD
{
    private int RandTiny => Random.Shared.Next(3, 8);
    private int RandSmall => Random.Shared.Next(31, 96);
    private int RandMedium => Random.Shared.Next(127, 384);
    private int RandLarge => Random.Shared.Next(785, 1536);

    [TestMethod]
    public void TestTinySEVDLower()
    => TestSVDCore(RandTiny, RandTiny);

    [TestMethod]
    public void TestSmallSEVDLower()
    => TestSVDCore(RandSmall, RandSmall);

    [TestMethod]
    public void TestMediumSEVDLower()
    => TestSVDCore(RandMedium, RandMedium);

    [TestMethod]
    public void TestLargeSEVDLower()
    => TestSVDCore(RandLarge, RandLarge);

    private static void TestSVDCore(int m, int n, double tol = 1e-10)
    {
        int count = 10;
        for (int i = 0; i < count; i++)
        {
            if(m < n)  (m, n) = (n, m);

            // Build a random symmetric matrix
            Matrix A = Matrix.RandomDense(m, n);
            Matrix orig = A.Clone();

            // Compute EVD: A = Q * D * Q^T
            var (U, svals, V) = new SVD(A);
            var D = Matrix.Diagonals(svals);

            // Reconstruction error
            var recon = U * (D * V.Transpose());
            var diff = orig - recon;
            double n1 = diff.Nrm1(),
            n2 = diff.NrmF(),
            nf = diff.NrmInf();

            // Orthogonality check: Q * Q^T ¡Ö I
            var I = Matrix.Eyes(m);
            var udiff = U * U.Transpose() - I;
            double un = udiff.NrmF();
            I = Matrix.Eyes(n);
            var vdiff = V * V.Transpose() - I;
            double vn = vdiff.NrmF();

            Assert.IsTrue(n1 < tol, $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol, $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol, $"NormInf error {nf} exceeds tolerance {tol}.");
            Assert.IsTrue(un < 1e-8, $"U orthogonality error {un} exceeds tolerance1e-8.");
            Assert.IsTrue(vn < 1e-8, $"V orthogonality error {vn} exceeds tolerance1e-8.");

            Console.WriteLine($"Test {i + 1}/{count}: n1 = {n1}, n2 = {n2}, nf = {nf}," +
                $" u = {un} v = {vn}");
        }
    }
}
