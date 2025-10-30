using DDLA.BLAS.Managed;
using DDLA.Core;
using DDLA.Transformations;
using DDLA.Utilities;
using System.Diagnostics;

using static DDLA.BLAS.Managed.BlasProvider;

namespace DDLA.Factorizations;

public class SVD
{
    readonly MatrixView matrix;
    bool computed;
    bool deconstructed;

    // results
    readonly Vector d;
    readonly Vector e;
    readonly Matrix U;
    readonly Matrix V;

    public SVD(MatrixView A, bool inplace = false)
    {
        var len = A.Rows;
        var wid = A.Cols;
        // Support both tall (m >= n) and wide (m < n) by allocating with min dimension
        matrix = inplace ? A : A.Clone();
        computed = false;
        U = Matrix.Eyes(len, colMajor: true);
        V = Matrix.Eyes(wid, colMajor: true);
        int k = Math.Min(len, wid);
        d = Vector.Create(k);
        e = Vector.Create(Math.Max(0, k - 1));
    }

    public Vector SingularValues
    {
        get
        {
            ComputeOnce();
            return d;
        }
    }

    public Matrix LeftSingularVectors
    {
        get
        {
            ComputeOnce();
            return U;
        }
    }

    public Matrix RightSingularVectors
    {
        get
        {
            ComputeOnce();
            return V;
        }
    }

    public void Deconstruct(out Matrix leftSingularVectors,
        out Matrix singularValues, out Matrix rightSingularVectors)
    {
        ComputeOnce();
        singularValues = Matrix.Create(U.Cols, V.Rows);
        singularValues.Diag.CopyFrom(d);
        leftSingularVectors = U;
        rightSingularVectors = V;

        deconstructed = true;
    }

    public void ComputeOnce()
    {
        if (deconstructed)
            throw new InvalidOperationException(
                "Matrix has been deconstructed, cannot compute again.");
        if (computed) return;
        computed = true;

        bool wide = matrix.Rows < matrix.Cols;

        if (!wide)
        {
            // Tall or square: A = U * B * V^T
            Bidiagonaling.Bidiag(matrix, U, V, d, e);
            var fran = new FrancisQRSVD(d, e, U, V);
            fran.Kernel();
        }
        else
        {
            // Wide: compute on A^T -> A^T = U_t * B * V_t^T
            // Map back: A = V_t * B^T * U_t^T, so original U <- V_t, V <- U_t
            Bidiagonaling.Bidiag(matrix.T, V, U, d, e);
            var fran = new FrancisQRSVD(d, e, V, U);
            fran.Kernel();
        }
    }
}

public class FrancisQRSVD(VectorView d,
    VectorView e, MatrixView U, MatrixView V,
    double tol = 1e-16, int maxIter = 400)
{
    public double Tol { get; private set; } = tol;

    public int MaxIter { get; } = maxIter * d.Length * d.Length;

    public int TotalIter { get; private set; } = 0;

    public VectorView d { get; } = d;

    public VectorView e { get; } = e;

    public MatrixView U { get; set; } = U;

    public MatrixView V { get; set; } = V;

    public void Kernel()
    {
        var len = d.Length * 2;
        using var _ = PoolUtils.Borrow<Givens>(len * 2, out var rots);
        var uRots = rots.AsSpan(0, len);
        var vRots = rots.AsSpan(len, len);
        ImplicitQrTridiag(d, e, U, V, uRots, vRots);
        MakeSingularValuesNonNegative();
        SortResults();
        Console.WriteLine($"Average calculate a singular value use {TotalIter / d.Length} transfers.");
    }

    public void ImplicitQrTridiag(VectorView d,
        VectorView e, MatrixView U, MatrixView V, Span<Givens> uRots, Span<Givens> vRots)
    {
        int m;
        int start, end;
        double tol = Tol * d.NrmF();

        if (d.Length <= 1)
        {
            return;
        }

        if (d.Length == 2)
        {
            ref var a = ref d[0];
            ref var b = ref e[0];
            ref var c = ref d[1];

            SVD2x2(d, e, tol, out var giv1, out var giv2);
            Rot(U[.., 0], U[.., 1], giv1);
            Rot(V[.., 0], V[.., 1], giv2);
            TotalIter++;
            return;
        }

        start = 0;
        end = d.Length - 1;

        while (start < end)
        {
            // 跳过已收敛的上下边界

            // 检查上边界
            for (m = end; m > start; m--)
            {
                var a = d[m - 1];
                var b = e[m - 1];
                var c = d[m];
                if (!Converged(a, b, c, tol))
                {
                    break;
                }
            }
            end = m;

            // 检查下边界
            for (m = start; m < end - 1; m++)
            {
                var a = d[m];
                var b = e[m];
                var c = d[m + 1];
                if (!Converged(a, b, c, tol))
                {
                    break;
                }
            }
            start = m;

            // 尝试寻找分割点
            m = start + 1;
            for (; m < end - 1; m++)
            {
                var a = d[m];
                var b = e[m];
                var c = d[m + 1];
                if (Converged(a, b, c, tol))
                {
                    break;
                }
            }
            // 如果找到了分割点，分裂矩阵
            if (m < end - 1)
            {
                // 矩阵分割，处理子问题
                // 左边：start..m右边：m..end
                //递归调用

                // var dStart = d.Offset;
                //Console.WriteLine(
                // $"[DEBUG]Split: {dStart + start}..{dStart + m} " +
                // $"and {dStart + m}..{dStart + end +1}");

                var d0 = d[start..(m + 1)];
                var e0 = e[start..m];
                var U0 = U[.., start..(m + 1)];
                var V0 = V[.., start..(m + 1)];
                var uRots0 = uRots[start..(m + 1)];
                var vRots0 = vRots[start..(m + 1)];

                var d1 = d[(m + 1)..(end + 1)];
                var e1 = e[(m + 1)..(end)];
                var U1 = U[.., (m + 1)..(end + 1)];
                var V1 = V[.., (m + 1)..(end + 1)];
                var uRots1 = uRots[(m + 1)..(end + 1)];
                var vRots1 = vRots[(m + 1)..(end + 1)];

                ImplicitQrTridiag(d0, e0, U0, V0, uRots0, vRots0);
                ImplicitQrTridiag(d1, e1, U1, V1, uRots1, vRots1);

                //由于调用结束后左右两部分均已收敛，
                // 因此当前对角化结束，直接返回
                break;
            }
            else
            {
                var len = end - start + 1;

                if (len <= 1)
                {
                    break;
                }
                else if (len == 2)
                {
                    var dWork = d[start..(end + 1)];
                    var eWork = e[start..end];
                    var UWork = U[.., start..(end + 1)];
                    var VWork = V[.., start..(end + 1)];
                    SVD2x2(dWork, eWork, tol,
                        out var giv1, out var giv2);
                    Rot(UWork[.., 0], UWork[.., 1], giv1);
                    Rot(VWork[.., 0], VWork[.., 1], giv2);
                    TotalIter++;
                    break;
                }
                else
                {
                    var dWork = d[start..(end + 1)];
                    var eWork = e[start..end];
                    var UWork = U[.., start..(end + 1)];
                    var VWork = V[.., start..(end + 1)];
                    var uRotsWork = uRots[start..end];
                    var vRotsWork = vRots[start..end];
                    BulgeStep(dWork, eWork,
                        //UWork, VWork, 
                        uRotsWork, vRotsWork);

                    var i = 0;
                    for (; i < eWork.Length - 1; i += 2)
                    {
                        var giv1 = uRotsWork[i];
                        var giv2 = uRotsWork[i + 1];
                        Rot2(UWork[.., i], UWork[.., i + 1], UWork[.., i + 2], giv1, giv2);
                    }
                    for (; i < eWork.Length - 0; i += 1)
                    {
                        var giv1 = uRotsWork[i];
                        Rot(UWork[.., i], UWork[.., i + 1], giv1);
                    }
                    if (i < eWork.Length)
                    {
                        var giv = uRotsWork[i];
                        Rot(UWork[.., i], UWork[.., i + 1], giv);
                    }

                    i = 0;
                    for (; i < 0 * eWork.Length; i += 2)
                    {
                        var giv1 = vRotsWork[i];
                        var giv2 = vRotsWork[i + 1];
                        Rot2(VWork[.., i], VWork[.., i + 1], VWork[.., i + 2], giv1, giv2);
                    }
                    for (; i < eWork.Length; i++)
                    {
                        var giv = vRotsWork[i];
                        Rot(VWork[.., i], VWork[.., i + 1], giv);
                    }
                    if (i < eWork.Length)
                    {
                        var giv = vRotsWork[i];
                        Rot(VWork[.., i], VWork[.., i + 1], giv);
                    }

                    TotalIter += dWork.Length;

                    var a = dWork[^2];
                    var b = eWork[^1];
                    var c = dWork[^1];
                    bool conv = Converged(a, b, c, tol);

                    if (!conv && TotalIter > MaxIter)
                    {
                        throw new Exception("Cannot Conv!");
                    }
                }
            }
        }
    }

    public void MakeSingularValuesNonNegative()
    {
        for (int i = 0; i < d.Length; i++)
        {
            if (d[i] < 0)
            {
                d[i] = -d[i];
                Scal(-1, V[.., i]);
            }
        }
    }

    void SortResults()
    {
        var d = this.d;
        int n = d.Length;
        if (n <= 1)
            return;
        while (d.Length > 1)
        {
            int maxIndex = 0;
            double maxValue = d[0];
            for (int i = 1; i < n; i++)
            {
                if (d[i] > maxValue)
                {
                    maxValue = d[i];
                    maxIndex = i;
                }
            }
            if (maxIndex != 0)
            {
                d[maxIndex] = d[0];
                d[0] = maxValue;
                U.SwapCol(0, maxIndex);
                V.SwapCol(0, maxIndex);
            }
            d = d[1..];
            U = U[.., 1..];
            V = V[.., 1..];
            n--;
        }
    }

    static bool Converged(double a, double b, double c, double tol)
    {
        return Math.Abs(b) <= tol * (Math.Abs(a) + Math.Abs(c)) + 1e-50;
    }

    static void SVD2x2(VectorView d, VectorView e, double tol,
        out Givens giv1, out Givens giv2)
    {
        ref var a = ref d[0];
        ref var b = ref e[0];
        ref var c = ref d[1];
        if (!Converged(a, b, c, tol))
        {
            SVD2x2Inner(a, b, c,
                out var cTmp, out var aTmp,
                out giv1, out giv2);
            a = aTmp;
            c = cTmp;
        }
        else
        {
            giv1 = giv2 = new(1, 0);
        }
    }

    // From LibFlame
    private static void SVD2x2Inner(in double alpha11,
                               in double alpha12,
                               in double alpha22,
                               out double sigma1,
                               out double sigma2,
                              out Givens giv1, out Givens giv2)
    {
        double zero = 0.0;
        double half = 0.5;
        double one = 1.0;
        double two = 2.0;
        double four = 4.0;

        double eps;

        double f, g, h;
        double clt = default, crt = default, slt = default, srt = default;
        double a, d, fa, ft, ga, gt, ha, ht, l;
        double m, mm, r, s, t, temp, tsign, tt;
        double ssmin = default, ssmax = default;
        double csl, snl;
        double csr, snr;

        bool gasmal;
        int pmax;
        bool swap;

        f = alpha11;
        g = alpha12;
        h = alpha22;

        eps = 1e-16;

        ft = f;
        fa = Math.Abs(f);
        ht = h;
        ha = Math.Abs(h);

        // pmax points to the maximum absolute element of matrix.
        // pmax =1 if f largest in absolute values.
        // pmax =2 if g largest in absolute values.
        // pmax =3 if h largest in absolute values.

        pmax = 1;

        swap = (ha > fa);
        if (swap)
        {
            pmax = 3;

            temp = ft;
            ft = ht;
            ht = temp;

            temp = fa;
            fa = ha;
            ha = temp;
        }

        gt = g;
        ga = Math.Abs(g);

        if (ga == 0.0)
        {
            // Diagonal matrix case.

            ssmin = ha;
            ssmax = fa;
            clt = one;
            slt = zero;
            crt = one;
            srt = zero;
        }
        else
        {
            gasmal = true;

            if (ga > fa)
            {
                pmax = 2;

                if ((fa / ga) < eps)
                {
                    // Case of very large ga.

                    gasmal = false;

                    ssmax = ga;

                    if (ha > one) ssmin = fa / (ga / ha);
                    else ssmin = (fa / ga) * ha;

                    clt = one;
                    slt = ht / gt;
                    crt = ft / gt;
                    srt = one;
                }
            }

            if (gasmal)
            {
                // Normal case.

                d = fa - ha;

                if (d == fa) l = one;
                else l = d / fa;

                m = gt / ft;

                t = two - l;

                mm = m * m;
                tt = t * t;
                s = Math.Sqrt(tt + mm);

                if (l == zero) r = Math.Abs(m);
                else r = Math.Sqrt(l * l + mm);

                a = half * (s + r);

                ssmin = ha / a;
                ssmax = fa * a;

                if (mm == zero)
                {
                    // Here, m is tiny.

                    if (l == zero) t = CopySign(two, ft) * CopySign(one, gt);
                    else t = gt / CopySign(d, ft) + m / t;
                }
                else
                {
                    t = (m / (s + t) + m / (r + l)) * (one + a);
                }

                l = Math.Sqrt(t * t + four);
                crt = two / l;
                srt = t / l;
                clt = (crt + srt * m) / a;
                slt = (ht / ft) * srt / a;
            }
        }

        if (swap)
        {
            csl = srt;
            snl = crt;
            csr = slt;
            snr = clt;
        }
        else
        {
            csl = clt;
            snl = slt;
            csr = crt;
            snr = srt;
        }


        // Correct the signs of ssmax and ssmin.

        if (pmax == 1)
            tsign = CopySign(one, csr) * CopySign(one, csl) * CopySign(one, f);
        else if (pmax == 2)
            tsign = CopySign(one, snr) * CopySign(one, csl) * CopySign(one, g);
        else // if ( pmax ==3 )
            tsign = CopySign(one, snr) * CopySign(one, snl) * CopySign(one, h);

        ssmax = CopySign(ssmax, tsign);
        ssmin = CopySign(ssmin, tsign * CopySign(one, f) * CopySign(one, h));

        // Save the output values.

        sigma1 = ssmin;
        sigma2 = ssmax;
        giv1 = (csl, snl);
        giv2 = (csr, snr);
    }

    public static double CopySign(double x, double y) =>
        y >= 0 ? x : -x;

    static Givens BulgeStep(VectorView d, VectorView e,
        //MatrixView U, MatrixView V, 
        Span<Givens> uRots, Span<Givens> vRots)
    {
        Debug.Assert(d.Length >= 3);
        Debug.Assert(d.Data != e.Data);
        int k = 0;
        //Console.WriteLine(
        // $"[DEBUG]BulgeStep on {d.Offset}..{d.Offset + d.Length}");

        double miu = WilkinsonShift(e[^2], d[^2], e[^1], d[^1]);
        // 如果 shift 有问题才 fallback
        if (double.IsNaN(miu) || miu < 0)

        {
            Console.WriteLine($"Warning: Invalid shift {miu}, using 0");
            miu = 0;
        }

        // CreateBulge(d, e, miu, out var bulge);
        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];
        var giv = ComputeGivens(d0 - miu, e0);

        d[0] = giv.c * d0 + giv.s * e0;
        e[0] = giv.c * e0 - giv.s * d0;
        d[1] = giv.c * d1;
        var bulge = giv.s * d1;

        vRots[k] = giv;

        while (d.Length >= 3)
        {
            // ChaseBulgeLeft(d, e, ref bulge);
            d0 = d[0];
            e0 = e[0];
            d1 = d[1];
            var e1 = e[1];

            giv = ComputeGivens(d0, bulge);

            Debug.Assert(Math.Abs(giv.c * bulge - giv.s * d0) < 1e-10);
            d[0] = giv.c * d0 + giv.s * bulge;
            e[0] = giv.c * e0 + giv.s * d1;
            d[1] = giv.c * d1 - giv.s * e0;
            e[1] = giv.c * e1;
            bulge = giv.s * e1;

            uRots[k] = giv;
            k++;

            //ChaseBulgeRight(d, e, ref bulge);
            e0 = e[0];
            d1 = d[1];
            e1 = e[1];
            var d2 = d[2];
            giv = ComputeGivens(e0, bulge);

            Debug.Assert(Math.Abs(giv.c * bulge - giv.s * e0) < 1e-10);
            e[0] = giv.c * e0 + giv.s * bulge;
            d[1] = giv.c * d1 + giv.s * e1;
            e[1] = giv.c * e1 - giv.s * d1;
            d[2] = giv.c * d2;
            bulge = giv.s * d2;

            vRots[k] = giv;

            d = d[1..];
            e = e[1..];
        }

        // DestroyBulge(d, e, in bulge);
        var d0f = d[0];
        var e0f = e[0];
        var d1f = d[1];

        giv = ComputeGivens(d0f, bulge);

        Debug.Assert(Math.Abs(giv.c * bulge - giv.s * d0f) < 1e-10);
        d[0] = giv.c * d0f + giv.s * bulge;
        e[0] = giv.c * e0f + giv.s * d1f;
        d[1] = giv.c * d1f - giv.s * e0f;
        uRots[k] = giv;
        return giv;
    }

    static Givens CreateBulge(VectorView d, VectorView e, double miu, out double bulge)
    {
        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];

        var giv = ComputeGivens(d0 - miu, e0);

        d[0] = d0 * giv.c + e0 * giv.s;
        e[0] = e0 * giv.c - d0 * giv.s;
        d[1] = d1 * giv.c;
        bulge = d1 * giv.s;

        return giv;
    }

    static Givens ChaseBulgeLeft(VectorView d, VectorView e, ref double bulge)
    {
        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];

        var giv = ComputeGivens(d0, bulge);

        Debug.Assert(Math.Abs(giv.c * bulge - giv.s * d0) < 1e-10);
        d[0] = giv.c * d0 + giv.s * bulge;
        e[0] = giv.c * e0 + giv.s * d1;
        d[1] = -giv.s * e0 + giv.c * d1;
        e[1] = giv.c * e1;
        bulge = giv.s * e1;

        return giv;
    }

    static Givens ChaseBulgeRight(VectorView d, VectorView e, ref double bulge)
    {
        // | e0 bl | | c s | 
        // | d1 e1 | | -s c |
        // |0 d2 |
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];
        var d2 = d[2];

        var giv = ComputeGivens(e0, bulge);

        // | e0 bl | | c s | = | e0 * c - bl * s e0 * s + bl * c |
        // | d1 e1 | | -s c | | d1 * c - e1 * s d1 * s + e1 * c |
        // |0 d2 | |0 * c - d2 * s0 * s + d2 * c |
        Debug.Assert(Math.Abs(-e0 * giv.s + bulge * giv.c) < 1e-10);
        e[0] = e0 * giv.c + bulge * giv.s;
        d[1] = d1 * giv.c + e1 * giv.s;
        e[1] = -d1 * giv.s + e1 * giv.c;
        d[2] = d2 * giv.c;
        bulge = d2 * giv.s;

        return giv;
    }

    static Givens DestroyBulge(VectorView d, VectorView e, in double bulge)
    {
        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];

        // | c -s | | d0 e0 |
        // | s c | | bl d1 |
        var giv = ComputeGivens(d0, bulge);

        // | c -s | | d0 e0 | = | c * d0 - s * bl c * e0 - s * d1 |
        // | s  c | | bl d1 | | s * d0 + c * bl s * e0 + c * d1 |
        Debug.Assert(Math.Abs(-d0 * giv.s + bulge * giv.c) < 1e-10);
        d[0] = giv.c * d0 + giv.s * bulge;
        e[0] = giv.c * e0 + giv.s * d1;
        d[1] = -giv.s * e0 + giv.c * d1;

        return giv;
    }

    private static void ApplyUV
        (VectorView left, VectorView right,
        Givens giv)
        => Rot(left, right, giv);

    private static void ApplyUV
        (VectorView left, VectorView right,
        double c, double s)
        => Rot(left, right, new(c, s));

    public static Givens ComputeGivens(double a, double b)
    {
        double tau, c, s;

        if (b == 0.0)
        {
            c = 1.0;
            s = 0.0;
        }
        else if (Math.Abs(b) > Math.Abs(a))
        {
            tau = a / b;
            s = Math.Sign(b) / Math.Sqrt(1.0 + tau * tau);
            c = s * tau;
        }
        else
        {
            tau = b / a;
            c = Math.Sign(a) / Math.Sqrt(1.0 + tau * tau);
            s = c * tau;
        }

        return new(c, s);
    }

    private static double WilkinsonShift(double a,
        double b, double c, double d)
    {
        var mat00 = a * a + b * b;
        var mat01 = b * c;
        var mat11 = c * c + d * d;
        var shift = WilkinsonShiftEVD(mat00, mat01, mat11);
        if (shift < 0)
        {
            return 0;
        }
        return Math.Sqrt(shift);
    }

    // From LibFlame
    private static double WilkinsonShiftEVD(double a,
        double b, double c)
    {
        // Compute a scaling factor to promote numerical stability.
        var scale = Math.Abs(a) + 2.0 * Math.Abs(b) + Math.Abs(c);

        if (scale == 0.0) return c;

        double invScale = 1 / scale;

        if (b != 0.0)
        {
            b *= invScale;

            double p = (invScale * a - invScale * c) / 2;
            double r = double.Hypot(p, b);

            if (a < c) p -= r;
            else p += r;

            c -= scale * (b * b / p);
        }
        return c;
    }
}

