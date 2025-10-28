using DDLA.Misc.Flags;
using System.Runtime.CompilerServices;
using DDLA.UFuncs.Operators;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;

using SIMDVec = System.Numerics.Vector<double>;
using SIMDExt = System.Numerics.Vector;
using DDLA.Utilities;
using DistIL.Attributes;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    /// <summary>
    /// y += Conj?(x).
    /// </summary>
    public static void Add(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Add(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    /// <summary>
    /// index = IndexOfMax(Abs(x)).
    /// </summary>
    /// <remarks><see cref="double.NaN"/> is seen as
    /// smaller than any other value.</remarks>
    public static int AMax(in vector x)
    {
        int length = x.Length;
        if (length == 0) return 0;
        Source.Amax(length, ref x.GetHeadRef(), x.Stride, out var indexDim);
        return (int)indexDim;
    }

    /// <summary>
    /// y += alpha * Conj?(x).
    /// </summary>
    public static void Axpy(scalar alpha, in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Axpy(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Axpby(scalar alpha, in vector x, scalar beta, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Axpby(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, in beta, ref y.GetHeadRef(), y.Stride);
    }

    public static void Copy(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Copy(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static scalar Dot(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return 0.0;
        Source.Dot(ConjType.NoConj, ConjType.NoConj, 
            length, 
            ref x.GetHeadRef(), x.Stride, 
            ref y.GetHeadRef(), y.Stride, 
            out double rho);
        return rho;
    }

    public static void Dotx(scalar alpha, in vector x, in vector y, scalar beta, ref scalar rho)
    {
        int length = CheckLength(x, y);
        if (length == 0)
        {
            rho *= beta;
            return;
        }
        Source.Dotx(ConjType.NoConj, ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride, in beta, ref rho);
    }

    public static void Invert(in vector x)
    {
        int length = x.Length;
        if (length == 0) return;
        Source.Invert(length, ref x.GetHeadRef(), x.Stride);
    }

    public static void InvScal(scalar alpha, in vector x)
    {
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        int length = x.Length;
        if (length == 0) return;
        Source.Scal(ConjType.NoConj, length, 1 / alpha, ref x.GetHeadRef(), x.Stride);
    }

    public static void Scal(scalar alpha, in vector x)
    {
        int length = x.Length;
        if (length == 0) return;
        Source.Scal(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride);
    }

    public static void Scal2(scalar alpha, in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Scal2(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Set(scalar alpha, in vector x)
    {
        int length = x.Length;
        if (length == 0) return;
        Source.Set(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride);
    }

    public static void Sub(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Sub(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Swap(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Swap(length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Xpby(in vector x, scalar beta, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Xpby(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, in beta, ref y.GetHeadRef(), y.Stride);
    }

    /// <summary>
    /// (x, y) = (c * x + s * y, - s * x + c * y)
    /// </summary>
    public static void Rot(in vector x, in vector y, Givens giv)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        if (giv.c == 1 && giv.s == 0) return;

        RotInner(in x, in y, giv.c, giv.s);
    }

    private static void RotInner(in vector x, in vector y, double c, double s)
    {
        int length = CheckLength(x, y);

        ref var xRef = ref x.GetHeadRef();
        ref var yRef = ref y.GetHeadRef();

        int i = 0;
        if (length > SIMDVec.Count * 4 &&
            x.Stride == 1 && y.Stride == 1)
        {
            var cVec = SIMDExt.Create(c);
            var sVec = SIMDExt.Create(s);
            var minuscVec = SIMDExt.Create(-s);
            var fma = new MultiplyAddOperator<double>();
            for (; i <= length - SIMDVec.Count; i += SIMDVec.Count)
            {
                var xVec = SIMDExt.LoadUnsafe(ref xRef);
                var yVec = SIMDExt.LoadUnsafe(ref yRef);
                var tmp1 = sVec * yVec;
                var tmp2 = minuscVec * xVec;
                xVec = fma.Invoke(in cVec, in xVec, in tmp1);
                yVec = fma.Invoke(in cVec, in yVec, in tmp2);
                SIMDExt.StoreUnsafe(xVec, ref xRef);
                SIMDExt.StoreUnsafe(yVec, ref yRef);
                xRef = ref Unsafe.Add(ref xRef, SIMDVec.Count);
                yRef = ref Unsafe.Add(ref yRef, SIMDVec.Count);
            }
        }
        for (; i <= length - 4; i += 4)
        {
            double xVal = xRef;
            double yVal = yRef;
            xRef = c * xVal + s * yVal;
            yRef = c * yVal - s * xVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            xVal = xRef;
            yVal = yRef;
            xRef = c * xVal + s * yVal;
            yRef = c * yVal - s * xVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            xVal = xRef;
            yVal = yRef;
            xRef = c * xVal + s * yVal;
            yRef = c * yVal - s * xVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            xVal = xRef;
            yVal = yRef;
            xRef = c * xVal + s * yVal;
            yRef = c * yVal - s * xVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
        }
        for (; i < length; i++)
        {
            double xVal = xRef;
            double yVal = yRef;
            xRef = c * xVal + s * yVal;
            yRef = c * yVal - s * xVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
        }
    }

    /// <summary>
    /// (x, y) = (c * x + s * y, - s * x + c * y)
    /// </summary>
    public static void Rot2(in vector x, in vector y, in vector z, Givens giv1, Givens giv2)
    {
        int length = CheckLength(x, y, z);
        if (length == 0) return;
        if (giv1.c == 1 && giv1.s == 0)
        {
            if (giv2.c != 1 || giv2.s != 0)
                RotInner(y, z, giv2.c, giv2.s);
        }
        else if (giv2.c == 1 && giv2.s == 0)
        {
            RotInner(x, y, giv1.c, giv1.s);
        }
        else
        {
            Rot2Inner(x, y, z, giv1.c, giv1.s, giv2.c, giv2.s);
        }
    }

    [Optimize]
    private static void Rot2Inner(in vector x, in vector y, in vector z, scalar c1, scalar s1,
        scalar c2, scalar s2)
    {
        int length = CheckLength(x, y);

        ref var xRef = ref x.GetHeadRef();
        ref var yRef = ref y.GetHeadRef();
        ref var zRef = ref z.GetHeadRef();

        int i = 0;
        if (length > SIMDVec.Count * 4 &&
            x.Stride == 1 && y.Stride == 1 && z.Stride == 1)
        {
            var c1Vec = SIMDExt.Create(c1);
            var s1Vec = SIMDExt.Create(s1);
            var minuss1Vec = SIMDExt.Create(-s1);
            var c2Vec = SIMDExt.Create(c2);
            var s2Vec = SIMDExt.Create(s2);
            var minuss2Vec = SIMDExt.Create(-s2);
            var fma = new MultiplyAddOperator<double>();
            for (; i <= length - SIMDVec.Count; i += SIMDVec.Count)
            {
                var xVec = SIMDExt.LoadUnsafe(ref xRef);
                var yVec = SIMDExt.LoadUnsafe(ref yRef);
                var tmp1 = s1Vec * yVec;
                var tmp2 = minuss1Vec * xVec;
                xVec = fma.Invoke(in c1Vec, in xVec, in tmp1);
                yVec = fma.Invoke(in c1Vec, in yVec, in tmp2);
                var zVec = SIMDExt.LoadUnsafe(ref zRef);
                tmp1 = s2Vec * zVec;
                tmp2 = minuss2Vec * yVec;
                yVec = fma.Invoke(in c2Vec, in yVec, in tmp1);
                zVec = fma.Invoke(in c2Vec, in zVec, in tmp2);
                SIMDExt.StoreUnsafe(xVec, ref xRef);
                SIMDExt.StoreUnsafe(yVec, ref yRef);
                SIMDExt.StoreUnsafe(zVec, ref zRef);
                xRef = ref Unsafe.Add(ref xRef, SIMDVec.Count);
                yRef = ref Unsafe.Add(ref yRef, SIMDVec.Count);
                zRef = ref Unsafe.Add(ref zRef, SIMDVec.Count);
            }
        }
        for (; i <= length - 4; i += 4)
        {
            double xVal = xRef;
            double yVal = yRef;
            double zVal = zRef;
            xRef = c1 * xVal + s1 * yVal;
            yRef = c1 * yVal - s1 * xVal;
            yVal = yRef;
            yRef = c2 * yVal + s2 * zVal;
            zRef = c2 * zVal - s2 * yVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            zRef = ref Unsafe.Add(ref zRef, z.Stride);

            xVal = xRef;
            yVal = yRef;
            zVal = zRef;
            xRef = c1 * xVal + s1 * yVal;
            yRef = c1 * yVal - s1 * xVal;
            yVal = yRef;
            yRef = c2 * yVal + s2 * zVal;
            zRef = c2 * zVal - s2 * yVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            zRef = ref Unsafe.Add(ref zRef, z.Stride);

            xVal = xRef;
            yVal = yRef;
            zVal = zRef;
            xRef = c1 * xVal + s1 * yVal;
            yRef = c1 * yVal - s1 * xVal;
            yVal = yRef;
            yRef = c2 * yVal + s2 * zVal;
            zRef = c2 * zVal - s2 * yVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            zRef = ref Unsafe.Add(ref zRef, z.Stride);

            xVal = xRef;
            yVal = yRef;
            zVal = zRef;
            xRef = c1 * xVal + s1 * yVal;
            yRef = c1 * yVal - s1 * xVal;
            yVal = yRef;
            yRef = c2 * yVal + s2 * zVal;
            zRef = c2 * zVal - s2 * yVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            zRef = ref Unsafe.Add(ref zRef, z.Stride);
        }
        for (; i < length; i++)
        {
            double xVal = xRef;
            double yVal = yRef;
            double zVal = zRef;
            xRef = c1 * xVal + s1 * yVal;
            yRef = c1 * yVal - s1 * xVal;
            yVal = yRef;
            yRef = c2 * yVal + s2 * zVal;
            zRef = c2 * zVal - s2 * yVal;
            xRef = ref Unsafe.Add(ref xRef, x.Stride);
            yRef = ref Unsafe.Add(ref yRef, y.Stride);
            zRef = ref Unsafe.Add(ref zRef, z.Stride);
        }
    }
}
