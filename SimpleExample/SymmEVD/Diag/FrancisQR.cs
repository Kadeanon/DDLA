using System.Diagnostics;
using System.Runtime.CompilerServices;
using DDLA.BLAS;
using DDLA.UFuncs.Operators;

namespace SimpleExample.SymmEVD.Diag;

public class FrancisQR(VectorView d,
    VectorView e, MatrixView? Q = null,
    double tol = 1e-16, int maxIter = 32)
{
    public double Tol { get; private set; } = tol;

    public int MaxIter { get; } = maxIter * d.Length * d.Length;

    public int TotalIter { get; private set; } = 0;

    public VectorView d { get; } = d;

    public VectorView e { get; } = e;

    public MatrixView? Q { get; set; } = Q;

    public void Kernel()
    {
        ImplicitQrTridiag(d, e, Q);
        SortResults();
    }

    void SortResults()
    {
        var d = this.d;
        int n = d.Length;
        if (n <= 1)
            return;
        if (this.Q is MatrixView Q)
        {
            while (d.Length > 1)
            {
                int minIndex = 0;
                double minValue = d[0];
                for (int i = 1; i < n; i++)
                {
                    if (d[i] < minValue)
                    {
                        minValue = d[i];
                        minIndex = i;
                    }
                }
                if (minIndex != 0)
                {
                    d[minIndex] = d[0];
                    d[0] = minValue;
                    Q.SwapCol(0, minIndex);
                }
                d = d[1..];
                Q = Q[.., 1..];
                n--;
            }
        }
        else
        {
            while (d.Length > 1)
            {
                int minIndex = 0;
                double minValue = d[0];
                for (int i = 1; i < n; i++)
                {
                    if (d[i] < minValue)
                    {
                        minValue = d[i];
                        minIndex = i;
                    }
                }
                if (minIndex != 0)
                {
                    d[minIndex] = d[0];
                    d[0] = minValue;
                }
                d = d[1..];
                n--;
            }
        }
    }

    public void ImplicitQrTridiag(VectorView d,
        VectorView e, MatrixView? QMaybe)
    {
        int m;
        int start, end;
        double tol = Tol * d.NrmF();

        if (d.Length <= 1)
        {
            TotalIter++;
            return;
        }

        if (d.Length == 2)
        {
            EVD2x2(d, e, tol, out var c, out var s);
            if (QMaybe is MatrixView Q)
                ApplyRight(Q[.., 0], Q[.., 1], c, s);
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
                    // 分割点为
                    e[m] = 0;
                    break;
                }
            }
            // 如果找到了分割点，分裂矩阵
            if (m < end - 1)
            {
                // 矩阵分割，处理子问题
                // 左边：start..m 右边：m..end
                // 递归调用
                var dStart = d.Offset;
                Console.WriteLine(
                    $"[DEBUG]Split: {dStart + start}..{dStart + m} " +
                    $"and {dStart + m}..{dStart + end + 1}");

                var d0 = d[start..(m + 1)];
                var e0 = e[start..m];
                var Q0 = QMaybe?[.., start..(m + 1)];

                var d1 = d[(m + 1)..(end + 1)];
                var e1 = e[(m + 1)..(end)];
                var Q1 = QMaybe?[.., (m + 1)..(end + 1)];

                ImplicitQrTridiag(d0, e0, Q0);
                ImplicitQrTridiag(d1, e1, Q1);

                // 由于调用结束后左右两部分均已收敛，
                // 因此当前对角化结束，直接返回
                break;
            }
            else
            {
                var len = end - start + 1;

                if (len <= 1)
                {
                    TotalIter++;
                }
                else if (len == 2)
                {
                    var dWork = d[start..(end + 1)];
                    var eWork = e[start..end];
                    EVD2x2(dWork, eWork, tol, out var c, out var s);
                    if (QMaybe is MatrixView Q)
                        ApplyRight(Q[.., start], Q[.., end], c, s);
                    TotalIter++;
                    break;
                }
                else
                {
                    var dWork = d[start..(end + 1)];
                    var eWork = e[start..end];
                    var QWork = QMaybe?[.., start..(end + 1)];
                    if (QWork is MatrixView Q)
                        BulgeStep(dWork, eWork, Q);
                    else
                        BulgeStep(dWork, eWork);

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

    static bool Converged(double a, double b, double c, double tol)
    {
        return Math.Abs(b) <= tol * (Math.Abs(a) + Math.Abs(c)) + 1e-100;
    }

    static void EVD2x2(VectorView d, VectorView e,
        double tol, out double g, out double s)
    {
        ref var a = ref d[0];
        ref var b = ref e[0];
        ref var c = ref d[1];
        if (Converged(a, b, c, tol))
        {
            g = 1;
            s = 0;
        }
        else
        {
            EVD2x2Inner(ref a, in b, ref c, 
                out g, out s);
            s = -s;
        }

    }

    // From LibFlame
    public static void EVD2x2Inner(ref double alpha11,
                                in double alpha21,
                                ref double alpha22,
                                out double gamma1,
                                out double sigma1)
    {
        double g1, s1;
        double acmn, acmx, acs, cs, ct, df, rt, sm, tb, tn;
        int sgn1, sgn2;

        // Compute the eigenvalues.

        sm = alpha11 + alpha22;
        df = alpha11 - alpha22;
        var dfs = df * df;
        tb = alpha21 + alpha21;
        var tbs = tb * tb;

        if (Math.Abs(alpha11) > Math.Abs(alpha22))
        {
            acmx = alpha11;
            acmn = alpha22;
        }
        else
        {
            acmx = alpha22;
            acmn = alpha11;
        }

        if (dfs > tbs)
            rt = Math.Abs(df) * Math.Sqrt(1.0 + tbs / dfs);
        else if (dfs < tbs)
            rt = Math.Abs(tb) * Math.Sqrt(1.0 + dfs / tbs);
        else
            rt = Math.Abs(tb) * Math.Sqrt(2.0);
        if (sm < 0.0)
        {
            alpha11 = 0.5 * (sm - rt);
            alpha22 = (acmx / alpha11) * acmn - (alpha21 / alpha11) * alpha21;
            sgn1 = -1;
        }
        else if (sm > 0.0)
        {
            alpha11 = 0.5 * (sm + rt);
            alpha22 = (acmx / alpha11) * acmn - (alpha21 / alpha11) * alpha21;
            sgn1 = 1;
        }
        else
        {
            alpha11 = 0.5 * rt;
            alpha22 = -0.5 * rt;
            sgn1 = 1;
        }

        // Compute the eigenvector.

        if (df >= 0.0)
        {
            cs = df + rt;
            sgn2 = 1;
        }
        else
        {
            cs = df - rt;
            sgn2 = -1;
        }

        acs = Math.Abs(cs);

        if (acs > Math.Abs(tb))
        {
            ct = -tb / cs;
            s1 = 1.0 / Math.Sqrt(1.0 + ct * ct);
            g1 = ct * s1;
        }
        else if (Math.Abs(tb) == 0.0)
        {
            g1 = 1.0;
            s1 = 0.0;
        }
        else
        {
            tn = -cs / tb;
            g1 = 1.0 / Math.Sqrt(1.0 + tn * tn);
            s1 = tn * g1;
        }

        if (sgn1 == sgn2)
        {
            gamma1 = -s1;
            sigma1 = g1;
        }
        else
        {
            gamma1 = g1;
            sigma1 = s1;
        }

    }

    static void BulgeStep(VectorView d, VectorView e)
    {
        Debug.Assert(d.Length >= 3);
        Debug.Assert(d.Data != e.Data);
        int n = d.Length;
        int k = 0;
        //Console.WriteLine(
        //    $"[DEBUG]BulgeStep on {d.Offset}..{d.Offset + d.Length}");

        double miu = WilkinsonShift
            (d[^2], e[^2], d[^1]);

        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];
        var d2 = d[2];

        ComputeGivens(d0 - miu, e0, out var c, out var s);

        var cc = c * c;
        var cs = c * s;
        var ss = s * s;

        d[0] = cc * d0 - cs * e0 - cs * e0 + ss * d1;
        e[0] = cs * d0 + cc * e0 - ss * e0 - cs * d1;
        d[1] = ss * d0 + cs * e0 + cs * e0 + cc * d1;
        var bulge = -s * e1;
        e[1] = c * e1;

        e0 = e[0];
        d1 = d[1];
        e1 = e[1];
        d2 = d[2];

        for (; k < n - 2; k++)
        {
            ComputeGivens(e0, bulge, out c, out s);

            cc = c * c;
            cs = c * s;
            ss = s * s;

            e[0] = c * e0 - s * bulge;
            d[1] = cc * d1 - cs * e1 - cs * e1 + ss * d2;
            e[1] = cs * d1 + cc * e1 - ss * e1 - cs * d2;
            d[2] = ss * d1 + cs * e1 + cs * e1 + cc * d2;

            if (k < n - 3)
            {
                ref var e2 = ref e[2];
                bulge = -s * e2;
                e2 *= c;

                e0 = e[1];
                d1 = d[2];
                e1 = e[2];
                d2 = d[3];

                d = d[1..];
                e = e[1..];
            }
        }
    }

    static void BulgeStep(VectorView d, VectorView e, MatrixView Q)
    {
        Debug.Assert(d.Length >= 3);
        Debug.Assert(d.Length == Q.Cols);
        Debug.Assert(d.Data != e.Data);
        int n = d.Length;
        int k = 0;
        //Console.WriteLine(
        //    $"[DEBUG]BulgeStep on {d.Offset}..{d.Offset + d.Length}");

        double miu = WilkinsonShift
            (d[^2], e[^2], d[^1]);

        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];
        var d2 = d[2];

        ComputeGivens(d0 - miu, e0, out var c, out var s);

        var cc = c * c;
        var cs = c * s;
        var ss = s * s;

        d[0] = cc * d0 - cs * e0 - cs * e0 + ss * d1;
        e[0] = cs * d0 + cc * e0 - ss * e0 - cs * d1;
        d[1] = ss * d0 + cs * e0 + cs * e0 + cc * d1;
        var bulge = -s * e1;
        e[1] = c * e1;

        e0 = e[0];
        d1 = d[1];
        e1 = e[1];
        d2 = d[2];

        ApplyRight(Q[.., 0], Q[.., 1], c, s);

        for (; k < n - 2; k++)
        {
            ComputeGivens(e0, bulge, out c, out s);

            cc = c * c;
            cs = c * s;
            ss = s * s;

            e[0] = c * e0 - s * bulge;
            d[1] = cc * d1 - cs * e1 - cs * e1 + ss * d2;
            e[1] = cs * d1 + cc * e1 - ss * e1 - cs * d2;
            d[2] = ss * d1 + cs * e1 + cs * e1 + cc * d2;
            ApplyRight(Q[.., 1], Q[.., 2], c, s);

            if (k < n - 3)
            {
                ref var e2 = ref e[2];
                bulge = -s * e2;
                e2 *= c;

                e0 = e[1];
                d1 = d[2];
                e1 = e[2];
                d2 = d[3];

                d = d[1..];
                e = e[1..];
                Q = Q[.., 1..];
            }
        }
    }

    static void StartBulge(VectorView d, VectorView e,
        double miu, out double bulge, 
        out double c, out double s)
    {
        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];

        ComputeGivens(d0 - miu,
            e0, out c, out s);

        var cc = c * c;
        var cs = c * s;
        var ss = s * s;

        d[0] = cc * d0 - cs * e0 - cs * e0 + ss * d1;
        e[0] = cs * d0 + cc * e0 - ss * e0 - cs * d1;
        d[1] = ss * d0 + cs * e0 + cs * e0 + cc * d1;
        bulge = - s * e1;
        e[1] = c * e1;

    }

    static void ChaseBulge(VectorView d, VectorView e,
        ref double bulge,
        out double c, out double s)
    {
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];
        var d2 = d[2];
        var e2 = e[2];

        ComputeGivens(e0,
            bulge, out c, out s);

        var cc = c * c;
        var cs = c * s;
        var ss = s * s;

        e[0] = c * e0 - s * bulge;
        d[1] = cc * d1 - cs * e1 - cs * e1 + ss * d2;
        e[1] = cs * d1 + cc * e1 - ss * e1 - cs * d2;
        d[2] = ss * d1 + cs * e1 + cs * e1 + cc * d2;

        e[2] = + c * e2;
        bulge = -s * e2;
    }

    static void DestroyBulge(VectorView d, VectorView e,
        in double bulge,
        out double c, out double s)
    {
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];
        var d2 = d[2];

        ComputeGivens(e0,
            bulge, out c, out s);

        var cc = c * c;
        var cs = c * s;
        var ss = s * s;

        e[0] = c * e0 - s * bulge;
        d[1] = cc * d1 - cs * e1 - cs * e1 + ss * d2;
        e[1] = cs * d1 + cc * e1 - ss * e1 - cs * d2;
        d[2] = ss * d1 + cs * e1 + cs * e1 + cc * d2;
    }

    private static void ApplyRight
        (VectorView left, VectorView right,
        double c, double s)
        => BlasProvider.Rot(left, right, c, -s);

    public static void ComputeGivens(double a, double b,
        out double c, out double s)
    {
        double tau;

        if (b == 0.0)
        {
            c = 1.0;
            s = 0.0;
        }
        else if (Math.Abs(b) > Math.Abs(a))
        {
            tau = -a / b;
            s = -Math.Sign(b) / Math.Sqrt(1.0 + tau * tau);
            c = s * tau;
        }
        else
        {
            tau = -b / a;
            c = Math.Sign(a) / Math.Sqrt(1.0 + tau * tau);
            s = c * tau;
        }
    }

    // From LibFlame
    private static double WilkinsonShift(double a, 
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

            return c - scale * (b * b / p);
        }
        else
        {
            return c;
        }
    }
}
