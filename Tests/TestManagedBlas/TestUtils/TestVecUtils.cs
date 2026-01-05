using BlasProvider = DDLA.BLAS.Managed.BlasProvider;
using BlisProvider = DDLA.BLAS.BlasProvider;

namespace Tests.TestManagedBlas.TestUtils;

[TestClass]
public class TestVecUtils
{
    internal static int length = Random.Shared.Next(48, 96);
    internal const int stride = 3;
    internal static double tol = Math.Sqrt(length) * 2e-16;

    [TestMethod]
    public void TestAsum()
    {
        var x = CreateVectorRandom(length);
        BlisProvider.Asum(x, out var blis_asum);
        BlasProvider.Asum(x, out var blas_asum);
        Assert.AreEqual(blis_asum, blas_asum, tol);

        x[1] = double.NaN;
        BlisProvider.Asum(x, out blis_asum);
        BlasProvider.Asum(x, out blas_asum);
        Assert.IsTrue(double.IsNaN(blis_asum) && double.IsNaN(blas_asum));

        x[1] = double.NegativeInfinity;
        BlisProvider.Asum(x, out blis_asum);
        BlasProvider.Asum(x, out blas_asum);
        Assert.IsTrue(double.IsInfinity(blis_asum) && double.IsInfinity(blas_asum));

        x = CreateVectorStrideRandom(length, stride);
        BlisProvider.Asum(x, out blis_asum);
        BlasProvider.Asum(x, out blas_asum);
        Assert.AreEqual(blis_asum, blas_asum, tol);

        x[1] = double.NaN;
        BlisProvider.Asum(x, out blis_asum);
        BlasProvider.Asum(x, out blas_asum);
        Assert.IsTrue(double.IsNaN(blis_asum) && double.IsNaN(blas_asum));

        x[1] = double.NegativeInfinity;
        BlisProvider.Asum(x, out blis_asum);
        BlasProvider.Asum(x, out blas_asum);
        Assert.IsTrue(double.IsInfinity(blis_asum) && double.IsInfinity(blas_asum));
    }

    [TestMethod]
    public void TestNrm1()
    {
        var x = CreateVectorRandom(length);
        var blis_nrm1 = BlisProvider.Nrm1(x);
        var blas_nrm1 = BlasProvider.Nrm1(x);
        Assert.AreEqual(blis_nrm1, blas_nrm1, tol);

        x[1] = double.NaN;
        blis_nrm1 = BlisProvider.Nrm1(x);
        blas_nrm1 = BlasProvider.Nrm1(x);
        Assert.IsTrue(double.IsNaN(blis_nrm1) && double.IsNaN(blas_nrm1));

        x[1] = double.NegativeInfinity;
        blis_nrm1 = BlisProvider.Nrm1(x);
        blas_nrm1 = BlasProvider.Nrm1(x);
        Assert.IsTrue(double.IsInfinity(blis_nrm1) && double.IsInfinity(blas_nrm1));

        x = CreateVectorStrideRandom(length, stride);
        blis_nrm1 = BlisProvider.Nrm1(x);
        blas_nrm1 = BlasProvider.Nrm1(x);
        Assert.AreEqual(blis_nrm1, blas_nrm1, tol);

        x[1] = double.NaN;
        blis_nrm1 = BlisProvider.Nrm1(x);
        blas_nrm1 = BlasProvider.Nrm1(x);
        Assert.IsTrue(double.IsNaN(blis_nrm1) && double.IsNaN(blas_nrm1));

        x[1] = double.NegativeInfinity;
        blis_nrm1 = BlisProvider.Nrm1(x);
        blas_nrm1 = BlasProvider.Nrm1(x);
        Assert.IsTrue(double.IsInfinity(blis_nrm1) && double.IsInfinity(blas_nrm1));
    }

    [TestMethod]
    public void TestNrmF()
    {
        var x = CreateVectorRandom(length);
        var blis_nrmF = BlisProvider.NrmF(x);
        var blas_nrmF = BlasProvider.NrmF(x);
        Assert.AreEqual(blis_nrmF, blas_nrmF, tol);

        x[1] = double.NaN;
        blis_nrmF = BlisProvider.NrmF(x);
        blas_nrmF = BlasProvider.NrmF(x);
        Assert.IsTrue(double.IsNaN(blis_nrmF) && double.IsNaN(blas_nrmF));

        x[1] = double.NegativeInfinity;
        blis_nrmF = BlisProvider.NrmF(x);
        blas_nrmF = BlasProvider.NrmF(x);
        Assert.IsTrue(double.IsInfinity(blis_nrmF) && double.IsInfinity(blas_nrmF));

        x = CreateVectorStrideRandom(length, stride);
        blis_nrmF = BlisProvider.NrmF(x);
        blas_nrmF = BlasProvider.NrmF(x);
        Assert.AreEqual(blis_nrmF, blas_nrmF, tol);

        x[1] = double.NaN;
        blis_nrmF = BlisProvider.NrmF(x);
        blas_nrmF = BlasProvider.NrmF(x);
        Assert.IsTrue(double.IsNaN(blis_nrmF) && double.IsNaN(blas_nrmF));

        x[1] = double.NegativeInfinity;
        blis_nrmF = BlisProvider.NrmF(x);
        blas_nrmF = BlasProvider.NrmF(x);
        Assert.IsTrue(double.IsInfinity(blis_nrmF) && double.IsInfinity(blas_nrmF));
    }

    [TestMethod]
    public void TestNrmInf()
    {
        var x = CreateVectorRandom(length);
        var blis_nrmInf = BlisProvider.NrmInf(x);
        var blas_nrmInf = BlasProvider.NrmInf(x);
        Assert.AreEqual(blis_nrmInf, blas_nrmInf, tol);

        x[1] = double.NaN;
        blis_nrmInf = BlisProvider.NrmInf(x);
        blas_nrmInf = BlasProvider.NrmInf(x);
        Assert.IsTrue(double.IsNaN(blis_nrmInf) && double.IsNaN(blas_nrmInf));

        x[1] = double.NegativeInfinity;
        blis_nrmInf = BlisProvider.NrmInf(x);
        blas_nrmInf = BlasProvider.NrmInf(x);
        Assert.IsTrue(double.IsInfinity(blis_nrmInf) && double.IsInfinity(blas_nrmInf));

        x = CreateVectorStrideRandom(length, stride);
        blis_nrmInf = BlisProvider.NrmInf(x);
        blas_nrmInf = BlasProvider.NrmInf(x);
        Assert.AreEqual(blis_nrmInf, blas_nrmInf, tol);

        x[1] = double.NaN;
        blis_nrmInf = BlisProvider.NrmInf(x);
        blas_nrmInf = BlasProvider.NrmInf(x);
        Assert.IsTrue(double.IsNaN(blis_nrmInf) && double.IsNaN(blas_nrmInf));

        x[1] = double.NegativeInfinity;
        blis_nrmInf = BlisProvider.NrmInf(x);
        blas_nrmInf = BlasProvider.NrmInf(x);
        Assert.IsTrue(double.IsInfinity(blis_nrmInf) && double.IsInfinity(blas_nrmInf));
    }
}
