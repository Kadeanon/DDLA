using DDLA.Einsum;

namespace Tests.TestEinsum
{
    [TestClass]
    public class TestBSMTC
    {
        #region double
        [TestMethod]
        public void TestTC()
        {
            int m = RandInt,
                n = RandIntShort, k = RandInt,
                c = RandInt, t = RandIntShort;
            Console.WriteLine($"m: {m}, n: {n}, k: {k}, c: {c}, t: {t}");
            Console.WriteLine($"M: {m}");
            Console.WriteLine($"N: {c}*{t} = {c * t}");
            Console.WriteLine($"K: {n}*{k} = {n * k}");
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mnk = MArray.Create([m, n, k]);
            MArray kntc = MArray.Create([k, n, t, c]);
            string expression = "mnk, kntc->mct";
            MArray mct = MArray.Contract(expression, mnk, kntc);

            var expect_mct = MArray.Create([m, c, t]);
            for (int m0 = 0; m0 < m; m0++)
            {
                for (int c0 = 0; c0 < c; c0++)
                {
                    Parallel.For(0, t, t0 =>
                    {
                        double res = mct[m0, c0, t0];
                        double exp = 0;
                        for (int n0 = 0; n0 < n; n0++)
                        {
                            for (int k0 = 0; k0 < k; k0++)
                            {
                                exp += mnk[m0, n0, k0] * kntc[k0, n0, t0, c0];
                            }
                        }
                        expect_mct[m0, c0, t0] = exp;
                        if (!eqa.Equals(exp, res))
                        {
                            Console.WriteLine($"{m0}, {c0}, {t0} exp: {exp}, got {res}");
                        }
                    });
                }
            }
            Assert.IsTrue(expect_mct.SequenceEqual(mct, eqa));
        }

        [TestMethod]
        public void TestTC_1LengthShouldBeRemove()
        {
            int m = RandInt,
                n = RandIntShort, k = RandInt,
                c = RandInt, t = RandIntShort,
                v = 1;
            Console.WriteLine($"m: {m}, n: {n}, k: {k}, c: {c}, t: {t}");
            Console.WriteLine($"M: {m}");
            Console.WriteLine($"N: {c}*{t} = {c * t}");
            Console.WriteLine($"K: {n}*{k} = {n * k}");
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mnvk = MArray.Create([m, n, v, k]);
            MArray kntc = MArray.Create([k, n, t, c]);
            string expression = "mnvk, kntc->mcvt";
            MArray mcvt =
                MArray.Contract(expression, mnvk, kntc);

            var expect_mcvt = MArray.Create([m, c, v, t]);
            for (int m0 = 0; m0 < m; m0++)
            {
                for (int c0 = 0; c0 < c; c0++)
                { 
                    for (int v0 = 0; v0 < v; v0++)
                    {
                        for (int t0 = 0; t0 < t; t0++)
                        {
                            double res = mcvt[m0, c0, v0, t0];
                            double exp = 0;
                            for (int n0 = 0; n0 < n; n0++)
                            {
                                for (int k0 = 0; k0 < k; k0++)
                                {
                                    exp += mnvk[m0, n0, v0, k0] * kntc[k0, n0, t0, c0];
                                }
                            }
                            expect_mcvt[m0, c0, v0, t0] = exp;
                            if (!eqa.Equals(exp, res))
                            {
                                Console.WriteLine($"{m0}, {c0}, {t0} exp: {exp}, got {res}");
                            }
                        }
                    }
                }
            }
            Assert.IsTrue(expect_mcvt.SequenceEqual(mcvt, eqa));
        }

        [TestMethod]
        public void TestMM()
        {
            int m = 1024 + 1,
                n = 256 + 1,// * RandInt, * RandInt,
                k = 256 + 1;// * RandInt;
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mk = MArray.Create([m, k]);
            MArray kn = MArray.Create([k, n]);
            string expression = "mk, kn->mn";
            MArray mn =
                MArray.Contract(expression, mk, kn);

            var expect_mn = MArray.Create([m, n]);
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

        [TestMethod]
        public void TestTCFast()
        {
            GEMMKernel ctl = new();
            int m = ctl.mc + ctl.mr,
                n = ctl.nc / 4,
                k = ctl.kc;
            double eps = 1e-10;
            var eqa = new AbsEqualComparer<double>(eps);
            MArray mk = MArray.Create([m, k]);
            MArray kn = MArray.Create([k, n]);
            string expression = "mk, kn->mn";
            MArray mn =
                MArray.Contract(expression, mk, kn);

            var expect_mn = MArray.Create([m, n]);
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
        #endregion

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
    }
}
