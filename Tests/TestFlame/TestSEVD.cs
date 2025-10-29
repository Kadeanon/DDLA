using DDLA.Core;
using DDLA.Factorizations;

namespace Tests.TestFlame;

[TestClass]
public class TestSEVD
{
    private const int tiny = 4;
    private const int small = 64;
    private const int medium = 256;
    private const int large =1024; // optional for perf

    [TestMethod]
    public void TestTinySEVDLower()
    => TestSEVDCore(tiny, UpLo.Lower);

    [TestMethod]
    public void TestSmallSEVDLower()
    => TestSEVDCore(small, UpLo.Lower);

    [TestMethod]
    public void TestMediumSEVDLower()
    => TestSEVDCore(medium, UpLo.Lower);

    [TestMethod]
    public void TestLargeSEVDLower()
    => TestSEVDCore(large, UpLo.Lower);

    [TestMethod]
    public void TestTinySEVDUpper()
    => TestSEVDCore(tiny, UpLo.Upper);

    [TestMethod]
    public void TestSmallSEVDUpper()
    => TestSEVDCore(small, UpLo.Upper);

    [TestMethod]
    public void TestMediumSEVDUpper()
    => TestSEVDCore(medium, UpLo.Upper);

    [TestMethod]
    public void TestLargeSEVDUpper()
    => TestSEVDCore(large, UpLo.Upper);

    private static void TestSEVDCore(int n, UpLo uplo, double tol = 1e-10)
    {
        int count = 10;
        for (int i = 0; i < count; i++)
        {
            // Build a random symmetric matrix
            Matrix A = Matrix.RandomSymmetric(n);
            if(uplo == UpLo.Upper)
                A = A.Transpose().Clone();
            Matrix orig = A.Clone();

            // Compute EVD: A = Q * D * Q^T
            var (evals, Q) = new SymmEVD(A);
            var D = Matrix.Diagonals(evals);

            // Reconstruction error
            var recon = Q * (D * Q.Transpose());
            var diff = orig - recon;
            double n1 = diff.Nrm1(),
            n2 = diff.NrmF(),
            nf = diff.NrmInf();

            // Orthogonality check: Q * Q^T ¡Ö I
            var I = Matrix.Eyes(n);
            var qdiff = Q * Q.Transpose() - I;
            double qn = qdiff.NrmF();

            Assert.IsTrue(n1 < tol, $"Norm1 error {n1} exceeds tolerance {tol}.");
            Assert.IsTrue(n2 < tol, $"Norm2 error {n2} exceeds tolerance {tol}.");
            Assert.IsTrue(nf < tol, $"NormInf error {nf} exceeds tolerance {tol}.");
            Assert.IsTrue(qn < 1e-8, $"Q orthogonality error {qn} exceeds tolerance1e-8.");

            Console.WriteLine($"Test {i + 1}/{count}: n1 = {n1}, n2 = {n2}, nf = {nf}, q = {qn}");
        }
    }
}
