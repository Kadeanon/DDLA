using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using DDLA.Misc;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using SIMDVec = System.Numerics.Vector<double>;
using SIMDExt = System.Numerics.Vector;
using static DDLA.UFuncs.UFunc;
using DDLA.Misc.Flags;
using DistIL.Attributes;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    /// <summary>
    /// y += Conj?(x).
    /// </summary>
    public static void Add(in vector x, in vector y)
        => x.Combine<AddOperator<scalar>>(y);

    /// <summary>
    /// y += alpha * Conj?(x).
    /// </summary>
    public static void Axpy(scalar alpha, in vector x, in vector y)
        => x.Combine<MultiplyAddOperator<scalar>, scalar>(alpha, y);

    /// <summary>
    /// y = alpha * Conj?(x) + beta * Conj?(y).
    /// </summary>
    public static void Axpby(in scalar alpha, in vector x, scalar beta, in vector y)
    {
        var indice = UFunc.CheckIndice(x, y);
        Details.AxpbyV_Impl(ref x[0], alpha, beta, ref y[0], indice);
    }

    public static partial class Details
    {
        public static void AxpbyV_Impl(ref scalar xHead, scalar alpha, scalar beta, 
            ref scalar yHead, DoubleIndice indice)
        {
            if ((indice.Length == 0) || (alpha == 0.0 && beta == 1.0))
                return;
            else if (alpha == 0.0)
                UFunc.Details.Map_Impl<MultiplyOperator<scalar>, scalar>
                    (ref yHead, beta, indice.B, default);
            else if (beta == 1.0)
                UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>
                    (ref xHead, alpha, ref yHead, indice, default);
            else if ((indice.AStride == 1) && (indice.BStride == 1))
                AxpbyV_Kernel_Vector256(indice.Length, ref xHead, alpha, beta, ref yHead);
            else
                AxpbyV_Kernel(ref xHead, alpha, beta, ref yHead, indice);
        }

        public static void AxpbyV_Kernel_Vector256(int length, ref scalar xHead, scalar alpha, scalar beta, ref scalar yHead)
        {
            int iterStride = 4;
            int iterSize = iterStride * Vector256<scalar>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 3);
                Vector256<scalar> alphaVec = Vector256.Create(alpha);
                Vector256<scalar> betaVec = Vector256.Create(beta);
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xHead);
                    Vector256<scalar> yVec0 = Vector256.LoadUnsafe(ref yHead);
                    Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xHead1);
                    Vector256<scalar> yVec1 = Vector256.LoadUnsafe(ref yHead1);
                    Vector256<scalar> xVec2 = Vector256.LoadUnsafe(ref xHead2);
                    Vector256<scalar> yVec2 = Vector256.LoadUnsafe(ref yHead2);
                    Vector256<scalar> xVec3 = Vector256.LoadUnsafe(ref xHead3);
                    Vector256<scalar> yVec3 = Vector256.LoadUnsafe(ref yHead3);
                    yVec0 *= betaVec;
                    yVec1 *= betaVec;
                    yVec2 *= betaVec;
                    yVec3 *= betaVec;
                    if (Fma.IsSupported)
                    {
                        yVec0 = Fma.MultiplyAdd(xVec0, alphaVec, yVec0);
                        yVec1 = Fma.MultiplyAdd(xVec1, alphaVec, yVec1);
                        yVec2 = Fma.MultiplyAdd(xVec2, alphaVec, yVec2);
                        yVec3 = Fma.MultiplyAdd(xVec3, alphaVec, yVec3);
                    }
                    else
                    {
                        yVec0 += xVec0 * alphaVec;
                        yVec1 += xVec1 * alphaVec;
                        yVec2 += xVec2 * alphaVec;
                        yVec3 += xVec3 * alphaVec;
                    }
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                    yVec0.StoreUnsafe(ref yHead);
                    yVec1.StoreUnsafe(ref yHead1);
                    yVec2.StoreUnsafe(ref yHead2);
                    yVec3.StoreUnsafe(ref yHead3);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                yHead *= beta;
                yHead += alpha * xHead;
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void AxpbyV_Kernel(ref scalar xHead, scalar alpha, scalar beta, ref scalar yHead, DoubleIndice indice)
        {
            int i = 0;
            for (; i <= indice.Length - 4; i += 4)
            {
                yHead *= beta;
                yHead += alpha * xHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                yHead *= beta;
                yHead += alpha * xHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                yHead *= beta;
                yHead += alpha * xHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                yHead *= beta;
                yHead += alpha * xHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
            for (; i < indice.Length; i++)
            {
                yHead *= beta;
                yHead += alpha * xHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    public static void Copy(in vector x, in vector y)
        => x.Map<IdentityOperator<scalar>>(y, default);

    public static scalar Dot(in vector x, in vector y)
        => ZipSum<MultiplyOperator<scalar>>(x, y);

    public static partial class Details
    {
        public static void DotV_Impl(ref scalar xHead, ref scalar yHead, DoubleIndice indice, out scalar rho)
        {
            if (indice.Length == 0)
                rho = 0.0;
            else if ((indice.AStride == 1) && (indice.BStride == 1))
                DotV_Kernel_Vector256(indice.Length, ref xHead, ref yHead, out rho);
            else
                DotV_Kernel(ref xHead, ref yHead, indice, out rho);
        }

        private static void DotV_Kernel_Vector256(int length, ref scalar xHead, ref scalar yHead, out scalar rho)
        {
            scalar rho2 = 0.0;
            int iterStride = 4;
            int iterSize = iterStride * Vector256<scalar>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 3);

                Vector256<scalar> rhoVec0 = Vector256<scalar>.Zero;
                Vector256<scalar> rhoVec1 = Vector256<scalar>.Zero;
                Vector256<scalar> rhoVec2 = Vector256<scalar>.Zero;
                Vector256<scalar> rhoVec3 = Vector256<scalar>.Zero;
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xHead);
                    Vector256<scalar> yVec0 = Vector256.LoadUnsafe(ref yHead);
                    Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xHead1);
                    Vector256<scalar> yVec1 = Vector256.LoadUnsafe(ref yHead1);
                    Vector256<scalar> xVec2 = Vector256.LoadUnsafe(ref xHead2);
                    Vector256<scalar> yVec2 = Vector256.LoadUnsafe(ref yHead2);
                    Vector256<scalar> xVec3 = Vector256.LoadUnsafe(ref xHead3);
                    Vector256<scalar> yVec3 = Vector256.LoadUnsafe(ref yHead3);
                    if (Fma.IsSupported)
                    {
                        rhoVec0 = Fma.MultiplyAdd(xVec0, yVec0, rhoVec0);
                        rhoVec1 = Fma.MultiplyAdd(xVec1, yVec1, rhoVec1);
                        rhoVec2 = Fma.MultiplyAdd(xVec2, yVec2, rhoVec2);
                        rhoVec3 = Fma.MultiplyAdd(xVec3, yVec3, rhoVec3);
                    }
                    else
                    {
                        rhoVec0 += xVec0 * yVec0;
                        rhoVec1 += xVec1 * yVec1;
                        rhoVec2 += xVec2 * yVec2;
                        rhoVec3 += xVec3 * yVec3;
                    }
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                }
                rhoVec0 += rhoVec1;
                rhoVec2 += rhoVec3;
                rhoVec0 += rhoVec2;
                rho2 += Vector256.Sum(rhoVec0);
            }
            for (; i < length; i++)
            {
                rho2 += xHead * yHead;
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
            rho = rho2;
        }

        public static void DotV_Kernel(ref scalar xHead, ref scalar yHead, DoubleIndice indice, out scalar rho)
        {
            rho = 0.0; 
            int i = 0;
            for (; i < indice.Length - 4; i += 4)
            {
                rho += xHead * yHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                rho += xHead * yHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                rho += xHead * yHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                rho += xHead * yHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
            for (; i < indice.Length; i++)
            {
                rho += xHead * yHead;
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    public static void Dotx(scalar alpha, in vector x, in vector y, scalar beta, ref scalar rho)
    {
        int length = CheckLength(x, y);
        rho *= beta;
        if (length == 0) return;
        rho += alpha * Dot(x, y);
    }

    public static void Invert(in vector x)
        => x.Map<DoubleInvertOperator>(new());

    public static void InvScal(scalar alpha, in vector x)
    {
        if (alpha == 0.0 || alpha == -0.0 || !scalar.IsFinite(alpha))
        {
            throw new ArgumentException(
                $"Error: alpha must be a finite non-zero value(got {alpha}).");
        }
        if(alpha != 1)
            x.Map<MultiplyOperator<scalar>, scalar>(1 / alpha);
    }

    public static void Scal(scalar alpha, in vector x)
    {
        if (alpha == 0.0 || alpha == -0.0)
            Set(alpha, x);
        else if(alpha != 1)
            x.Map<MultiplyOperator<scalar>, scalar>(alpha);
    }

    public static void Scal2(scalar alpha, in vector x, in vector y)
    {
        if (alpha == 0.0 || alpha == -0.0)
            Set(alpha, y);
        else if(alpha != 1)
            x.Map<MultiplyOperator<scalar>, scalar>(alpha, y, default);
        else
            Copy(x, y);
    }

    public static void Pow(scalar alpha, in vector x)
        => x.Map<PowOperator<scalar>, scalar>(alpha);

    public static void Sqrt(in vector x)
        => x.Map<SqrtOperator<scalar>>();

    public static void Set(scalar alpha, in vector x)
        => x.Apply<IdentityOperator<scalar>, scalar>(alpha);

    public static void Sub(in vector x, in vector y)
        => x.Combine<ReversedOp<SubtractOperator<scalar>, scalar, scalar, scalar>>(y);

    public static void Swap(in vector x, in vector y)
    {
        var indice = UFunc.CheckIndice(x, y);
        Details.SwapV_Impl(ref x[0], ref y[0], indice);
    }

    public static partial class Details
    {
        public static void SwapV_Impl(ref scalar xHead, 
            ref scalar yHead, DoubleIndice indice)
        {
            if (indice.Length == 0)
                return;
            else if ((indice.AStride == 1) && (indice.BStride == 1))
                SwapV_Kernel_Vector256(indice.Length, ref xHead, ref yHead);
            else
                SwapV_Kernel(ref xHead, ref yHead, indice);
        }

        private static void SwapV_Kernel_Vector256(int length, ref scalar xHead, ref scalar yHead)
        {
            int iterStride = 4;
            int iterSize = iterStride * Vector256<scalar>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 4);
                ref var yHead1 = ref Unsafe.Add(ref yHead, 4);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 8);
                ref var yHead2 = ref Unsafe.Add(ref yHead, 8);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 12);
                ref var yHead3 = ref Unsafe.Add(ref yHead, 12);
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xHead);
                    Vector256<scalar> yVec0 = Vector256.LoadUnsafe(ref yHead);
                    Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xHead1);
                    Vector256<scalar> yVec1 = Vector256.LoadUnsafe(ref yHead1);
                    Vector256<scalar> xVec2 = Vector256.LoadUnsafe(ref xHead2);
                    Vector256<scalar> yVec2 = Vector256.LoadUnsafe(ref yHead2);
                    Vector256<scalar> xVec3 = Vector256.LoadUnsafe(ref xHead3);
                    Vector256<scalar> yVec3 = Vector256.LoadUnsafe(ref yHead3);
                    Vector256.StoreUnsafe(xVec0, ref yHead);
                    Vector256.StoreUnsafe(xVec1, ref yHead1);
                    Vector256.StoreUnsafe(xVec2, ref yHead2);
                    Vector256.StoreUnsafe(xVec3, ref yHead3);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                    Vector256.StoreUnsafe(yVec0, ref xHead);
                    Vector256.StoreUnsafe(yVec1, ref xHead1);
                    Vector256.StoreUnsafe(yVec2, ref xHead2);
                    Vector256.StoreUnsafe(yVec3, ref xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                (xHead, yHead) = (yHead, xHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void SwapV_Kernel(ref scalar xHead, ref scalar yHead, DoubleIndice indice)
        {
            int i = 0;
            for (; i < indice.Length - 4; i += 4)
            {
                (xHead, yHead) = (yHead, xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                (xHead, yHead) = (yHead, xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                (xHead, yHead) = (yHead, xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
                (xHead, yHead) = (yHead, xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
            for (; i < indice.Length; i++)
            {
                (xHead, yHead) = (yHead, xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    public static void Shift(scalar alpha, in vector x)
        => x.Map<AddOperator<scalar>, scalar>(alpha);

    public static void Xpby(in vector x, scalar beta, in vector y)
        => x.Combine<DoubleXpbyOperator, scalar>(beta, y);

    public static scalar RMS(in vector x)
        => x.Length == 0 ? 0.0 : 
        Math.Sqrt(Sum<SquareOperator<scalar>>(x) / x.Length);

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
    /// (x, y) = (c * x + s * y, - s * x + c * y)
    /// </summary>
    public static void Rot(in vector x, in vector y, Givens giv)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        if (giv.c == 1 && giv.s == 0) return;

        RotInner(in x, in y, giv.c, giv.s);
    }

    //[Optimize]
    internal unsafe static void RotInner(in vector x, in vector y, double c, double s)
    {
        int length = CheckLength(x, y);

        ref var xRef = ref x.GetHeadRef();
        ref var yRef = ref y.GetHeadRef();

        int i = 0;
        if (length > SIMDVec.Count * 4 &&
            x.Stride == 1 && y.Stride == 1)
        {
            for (var len = 0; len < 32; len += 4)
            {
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref xRef, SIMDVec.Count * len)));
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref yRef, SIMDVec.Count * len)));
            }
            var cVec = SIMDExt.Create(c);
            var sVec = SIMDExt.Create(s);
            var minuscVec = SIMDExt.Create(-s);
            var fma = new MultiplyAddOperator<double>();
            for (; i <= length - SIMDVec.Count; i += SIMDVec.Count)
            {
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref xRef, SIMDVec.Count * 32)));
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref yRef, SIMDVec.Count * 32)));
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

    //[Optimize]
    internal unsafe static void Rot2Inner(in vector x, in vector y, in vector z, scalar c1, scalar s1,
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
            for (var len = 0; len < 32; len += 4)
            {
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref xRef, SIMDVec.Count * len)));
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref yRef, SIMDVec.Count * len)));
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref zRef, SIMDVec.Count * len)));
            }
            var c1Vec = SIMDExt.Create(c1);
            var s1Vec = SIMDExt.Create(s1);
            var minuss1Vec = SIMDExt.Create(-s1);
            var c2Vec = SIMDExt.Create(c2);
            var s2Vec = SIMDExt.Create(s2);
            var minuss2Vec = SIMDExt.Create(-s2);
            var fma = new MultiplyAddOperator<double>();
            for (; i <= length - SIMDVec.Count; i += SIMDVec.Count)
            {
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref xRef, SIMDVec.Count * 32)));
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref yRef, SIMDVec.Count * 32)));
                Sse.Prefetch1(Unsafe.AsPointer(ref Unsafe.Add(ref zRef, SIMDVec.Count * 32)));
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

    private static TripleIndice CheckIndice(in vector x, in vector y, in vector z)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Error: x and y must have the same length.");
        if (x.Length != z.Length)
            throw new ArgumentException("Error: x and z must have the same length.");
        return new(x.Length, x.Stride, y.Stride, z.Stride);
    }

}
