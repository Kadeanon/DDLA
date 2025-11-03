using DDLA.Einsum;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Tests.TestEinsum
{
    [TestClass]
    public class TestBSMTC
    {
        [TestMethod]
        public void TestTC()
        {
            int m = RandIntShort,
                n = RandIntShort, k = RandIntShort,
                c = RandIntShort, t = RandIntShort;
            Console.WriteLine($"m: {m}, n: {n}, k: {k}, c: {c}, t: {t}");
            Console.WriteLine($"M: {m}");
            Console.WriteLine($"N: {c}*{t} = {c * t}");
            Console.WriteLine($"K: {n}*{k} = {n * k}");
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mnk = CreateRandom([m, n, k]);
            MArray nktc = CreateRandom([n, k, t, c]);
            string expression = "mnk, nktc->mct";
            MArray mct = MArray.Contract(expression, mnk, nktc);

            var expect_mct = CreateRandom([m, c, t]);
            for (int m0 = 0; m0 < m; m0++)
            {
                for (int c0 = 0; c0 < c; c0++)
                {
                    for (int t0 = 0; t0 < t; t0++)
                    {
                        double res = mct[m0, c0, t0];
                        double exp = 0;
                        for (int n0 = 0; n0 < n; n0++)
                        {
                            for (int k0 = 0; k0 < k; k0++)
                            {
                                exp += mnk[m0, n0, k0] * nktc[n0, k0, t0, c0];
                            }
                        }
                        expect_mct[m0, c0, t0] = exp;
                        if (!eqa.Equals(exp, res))
                        {
                            Console.WriteLine($"{m0}, {c0}, {t0} exp: {exp}, got {res}");
                        }
                    }
                }
            }
            Assert.IsTrue(expect_mct.SequenceEqual(mct, eqa));
        }

        [TestMethod]
        public void TestMM()
        {
            int m = 1024 + 1,
                n = 256 + 1,// * RandInt, * RandInt,
                k = 256 + 1;// * RandInt;
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mk = CreateRandom([m, k]);
            MArray kn = CreateRandom([k, n]);
            string expression = "mk, kn->mn";
            MArray mn =
                MArray.Contract(expression, mk, kn);

            var expect_mn = CreateRandom([m, n]);
            var mk_mat = mk.AsMatrix();
            var kn_mat = kn.AsMatrix();
            var mn_expected = MArray.FromMatrix(mk_mat * kn_mat);

            for (int m0 = 0; m0 < m; m0++)
            {
                for (int n0 = 0; n0 < n; n0++)
                {
                    var res = mn[m0, n0];
                    var exp = mn_expected[m0, n0];
                    if (!eqa.Equals(exp, res))
                    {
                        Console.WriteLine($"{m0}, {n0}, exp: {exp}, got {res}");
                    }
                }
            }
        }

        [TestMethod]
        public void TestMMFast()
        {
            GEMMKernel ctl = new();
            int m = ctl.mc + 2,
                n = ctl.nc + 1,
                k = ctl.kc;
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mk = CreateRandom([m, k]);
            MArray kn = CreateRandom([k, n]);
            string expression = "mk, kn->mn";
            MArray mn =
                MArray.Contract(expression, mk, kn);

            var expect_mn = CreateRandom([m, n]);
            for (int m0 = 0; m0 < m; m0++)
            {
                for (int n0 = 0; n0 < n; n0++)
                {
                    double res = mn[m0, n0];
                    double exp = 0;
                    for (int k0 = 0; k0 < k; k0++)
                    {
                        exp += mk[m0, k0] * kn[k0, n0];
                    }
                    expect_mn[m0, n0] = exp;
                    if (!eqa.Equals(exp, res))
                    {
                        Console.WriteLine($"{m0}, {n0}, exp: {exp}, got {res}");
                    }
                }
            }
            Assert.IsTrue(expect_mn.SequenceEqual(mn, eqa));
        }

        [Ignore("need to fix")]
        [TestMethod]
        public void TestTC1v1()
        {
            GEMMKernel ctl = new();
            int m = ctl.mc + ctl.mr;
            double eps = 1e-10;
            MArray a = CreateRandom([m]);
            MArray b = CreateRandom([m]);
            string expression = "m, m->";
            MArray c =
                MArray.Contract(expression, a, b);
            Assert.AreEqual(1, c.Rank);
            Assert.AreEqual(1, c.Lengths[0]);
            double exp = 0;
            for (int m0 = 0; m0 < m; m0++)
            {
                exp += a[m0] * b[m0];
            }
            Assert.AreEqual(exp, c[0], eps);
        }

        public int RandInt
        {
            get
            {
                var r = Random.Shared;
                return (int)r.NextInt64(65, 130);
            }
        }

        public int RandIntShort
        {
            get
            {
                var r = Random.Shared;
                return (int)r.NextInt64(14, 37);
            }
        }

        public byte RandOne
        {
            get
            {
                var r = Random.Shared;
                return (byte)(r.NextInt64() & 2);
            }
        }

        public MArray CreateRandom(params ReadOnlySpan<int> dims)
        {
            int totalLength = 1;
            foreach (var dim in dims)
            {
                totalLength *= dim;
            }
            double[] data = new double[totalLength];
            var r = Random.Shared;
            for (int i = 0; i < totalLength; i++)
            {
                data[i] = r.NextDouble();
            }
            return new MArray(data, dims.ToArray());
        }
    }
}
