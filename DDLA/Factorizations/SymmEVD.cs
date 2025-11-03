using DDLA.BLAS.Managed;
using DDLA.Core;
using DDLA.Misc.Flags;
using DDLA.Transformations;
using DDLA.Utilities;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DDLA.Factorizations;

/// <summary>
/// Symmetric Eigenvalue Decomposition (EVD) for real symmetric matrices.
/// Reduces A to tridiagonal form (Householder UT), then applies Francis implicit QR on T.
/// Returns A = Q * diag(d) * Q^T with sorted eigenvalues (ascending).
/// </summary>
public class SymmEVD
{
    readonly Matrix matrix; // working matrix (may be clone of input)
    bool computed;
    bool deconstructed;

    // results
    Vector d; // eigenvalues (length n)
    Vector e; // subdiagonal (length n-1)
    Matrix Q; // eigenvectors (n x n)

    public SymmEVD(MatrixView A, UpLo uplo = UpLo.Lower, bool inplace = false)
    {
        // Only the specified triangular part is referenced.
        BlasProvider.CheckSymmMatLength(A, uplo);
        if (uplo == UpLo.Upper)
            A = A.T;
        matrix = inplace ? new(A) : A.Clone();
        computed = false;
        Q = matrix;
        d = Vector.Create(matrix.Rows);
        e = Vector.Create(matrix.Rows - 1);
    }

    public SymmEVD(Matrix A, UpLo uplo = UpLo.Lower, bool inplace = false)
    {
        BlasProvider.CheckSymmMatLength(A, uplo);
        if (uplo == UpLo.Upper)
        {
            matrix = inplace ? A.Transpose() : A.View.T.Clone();
        }
        else
        {
            matrix = inplace ? A : A.Clone();
        }
        computed = false;
        Q = matrix;
        d = Vector.Create(matrix.Rows);
        e = Vector.Create(matrix.Rows - 1);
    }

    public Vector EigenValues
    {
        get
        {
            ComputeOnce();
            return d;
        }
    }

    public Matrix EigenVectors
    {
        get
        {
            ComputeOnce();
            return Q;
        }
    }

    public void Deconstruct(out Vector eigenValues, out Matrix eigenVectors)
    {
        ComputeOnce();
        eigenValues = d;
        eigenVectors = Q;

        deconstructed = true;
    }

    public void ComputeOnce()
    {
        if (deconstructed)
            throw new InvalidOperationException(
                "Matrix has been deconstructed, cannot compute again.");
        if (computed) return;
        computed = true;

        var T = Tridiagonaling.CreateT(matrix);
        // Reduce to tridiagonal and form Q in-place into matrix (as per Tridiagonaling).
        Tridiagonaling.Tridiag(matrix, T, d, e);
        // After Tridiagonaling.Tridiag, matrix now stores the orthogonal Q in full.
        Q = matrix;

        // Run implicit-shift QR for tridiagonal eigenvalues/vectors.
        var fran = new FrancisQRSEVD(d, e, Q);
        fran.Kernel();
        // d is now sorted ascending; Q columns are the corresponding eigenvectors.
    }
}

/// <summary>
/// Francis implicit QR algorithm on symmetric tridiagonal matrix.
/// Operates in-place on main diagonal d and subdiagonal e, and accumulates
/// the orthogonal similarity transforms into Q (columns are eigenvectors).
/// On return, d contains sorted eigenvalues (ascending), and Q is reordered accordingly.
/// </summary>
public class FrancisQRSEVD(VectorView d,
    VectorView e, MatrixView Q,
    double tol = 1e-16, int maxIter = 32)
{
    public double Tol { get; private set; } = tol;

    public int MaxIter { get; } = maxIter * d.Length * d.Length;

    public int TotalIter { get; private set; } = 0;

    public VectorView d { get; } = d;

    public VectorView e { get; } = e;

    public MatrixView Q { get; set; } = Q;

    public void Kernel()
    {
        using var _ = PoolUtils.Borrow<Givens>(e.Length, out var rots);
        ImplicitQrTridiag(d, e, Q, rots);
        SortResults();
        //Console.WriteLine($"Average calculate a eigenValue use {TotalIter / d.Length} sweep.");
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
        VectorView e, MatrixView Q, Span<Givens> rots)
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
            EVD2x2(d, e, tol, out var giv);
            ApplyRight(Q[.., 0], Q[.., 1], giv);
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
                //var dStart = d.Offset;
                //Console.WriteLine(
                //    $"[DEBUG]Split: {dStart + start}..{dStart + m} " +
                //    $"and {dStart + m}..{dStart + end + 1}");

                var d0 = d[start..(m + 1)];
                var e0 = e[start..m];
                var Q0 = Q[.., start..(m + 1)];
                var rot0 = rots[start..m];

                var d1 = d[(m + 1)..(end + 1)];
                var e1 = e[(m + 1)..(end)];
                var Q1 = Q[.., (m + 1)..(end + 1)];
                var rot1 = rots[(m + 1)..end];

                ImplicitQrTridiag(d0, e0, Q0, rot0);
                ImplicitQrTridiag(d1, e1, Q1, rot1);

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
                    EVD2x2(dWork, eWork, tol, out var giv);
                    ApplyRight(Q[.., start], Q[.., end], giv);
                    TotalIter++;
                    break;
                }
                else
                {
                    var dWork = d[start..(end + 1)];
                    var eWork = e[start..end];
                    var QWork = Q[.., start..(end + 1)];
                    var rotWork = rots[start..end];
                    BulgeStep(dWork, eWork, rotWork);


                    var i = 0;
                    for (; i < eWork.Length - 1; i += 2)
                    {
                        var (c1, s1) = rotWork[i];
                        var (c2, s2) = rotWork[i + 1];
                        BlasProvider.Rot2(QWork[.., i], QWork[.., i + 1], QWork[.., i + 2],
                            (c1, -s1), (c2, -s2));
                    }
                    for (; i < eWork.Length; i++)
                    {
                        var (c1, s1) = rotWork[i];
                        BlasProvider.Rot(QWork[.., i], QWork[.., i + 1], (c1, -s1));
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

    static bool Converged(double a, double b, double c, double tol)
    {
        return Math.Abs(b) <= tol * (Math.Abs(a) + Math.Abs(c)) + 1e-100;
    }

    static void EVD2x2(VectorView d, VectorView e,
        double tol, out Givens giv)
    {
        ref var a = ref d[0];
        ref var b = ref e[0];
        ref var c = ref d[1];
        if (Converged(a, b, c, tol))
        {
            giv = (1, 0);
        }
        else
        {
            EVD2x2Inner(ref a, in b, ref c,
                out var g, out var s);
            giv = (g, -s);
        }

    }

    // From LibFlame
    public static void EVD2x2Inner(ref double a,
                                in double b,
                                ref double c,
                                out double g,
                                out double s)
    {
        double min, max, acs, cs, delta;
        bool low, high;

        // Compute the eigenvalues.

        var sum = a + c;
        var diff = a - c;
        var diffSq = diff * diff;
        var b2 = b + b;
        var b2sq = b2 * b2;

        if (Math.Abs(a) > Math.Abs(c))
        {
            max = a;
            min = c;
        }
        else
        {
            max = c;
            min = a;
        }

        if (diffSq > b2sq)
            delta = Math.Abs(diff) * Math.Sqrt(1.0 + b2sq / diffSq);
        else if (diffSq < b2sq)
            delta = Math.Abs(b2) * Math.Sqrt(1.0 + diffSq / b2sq);
        else
            delta = Math.Abs(b2) * Math.Sqrt(2.0);

        if (sum == 0.0)
        {
            a = 0.5 * delta;
            c = -0.5 * delta;
            low = false;
        }
        else if (sum < 0.0)
        {
            a = 0.5 * (sum - delta);
            c = (max / a) * min - (b / a) * b;
            low = true;
        }
        else
        {
            a = 0.5 * (sum + delta);
            c = (max / a) * min - (b / a) * b;
            low = false;
        }

        // Compute the eigenvector.

        if (a >= c)
        {
            cs = diff + delta;
            high = true;
        }
        else
        {
            cs = diff - delta;
            high = false;
        }

        acs = Math.Abs(cs);
        var atb = Math.Abs(b2);

        double tau;
        if (atb == 0.0)
        {
            g = 1.0;
            s = 0.0;
        }
        else if (acs > atb)
        {
            tau = -b2 / cs;
            s = 1.0 / double.Hypot(1.0, tau);
            g = tau * s;
        }
        else
        {
            tau = -cs / b2;
            g = 1.0 / double.Hypot(1.0, tau);
            s = tau * g;
        }

        if (low != high)
            (g, s) = (-s, g);
    }

    static void BulgeStep(VectorView d, VectorView e,
        Span<Givens> rots)
    {
        Debug.Assert(d.Length >= 3);
        Debug.Assert(d.Data != e.Data);
        int n = d.Length;
        int k = 0;
        //Console.WriteLine(
        //    $"[DEBUG]BulgeStep on {d.Offset}..{d.Offset + d.Length}");

        double miu = WilkinsonShift
            (d[^2], e[^1], d[^1]);
        ref var rot = ref rots[0];

        var d0 = d[0];
        var e0 = e[0];
        var d1 = d[1];
        var e1 = e[1];
        var d2 = d[2];

        ComputeGivens(d0 - miu, e0, out var c, out var s);
        rot = (c, s);

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
            rot = ref Unsafe.Add(ref rot, 1);
            ComputeGivens(e0, bulge, out c, out s);

            cc = c * c;
            cs = c * s;
            ss = s * s;

            e[0] = c * e0 - s * bulge;
            d[1] = cc * d1 - cs * e1 - cs * e1 + ss * d2;
            e[1] = cs * d1 + cc * e1 - ss * e1 - cs * d2;
            d[2] = ss * d1 + cs * e1 + cs * e1 + cc * d2;
            rot = (c, s);

            if (k != n - 3)
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

    private static void ApplyRight
        (VectorView left, VectorView right,
        double c, double s)
        => BlasProvider.Rot(left, right, (c, -s));

    private static void ApplyRight
        (VectorView left, VectorView right,
        Givens giv)
        => BlasProvider.Rot(left, right, (giv.c, -giv.s));

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

