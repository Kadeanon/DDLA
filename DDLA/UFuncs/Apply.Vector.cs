using DDLA.Core;
using DDLA.Misc;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Vector = System.Numerics.Vector;

namespace DDLA.UFuncs;

public static partial class UFunc
{
    public static TOperator OrDefault<TOperator>(this TOperator? input)
        where TOperator : struct, IOperator
    {
        return input ?? (TOperator.ShouldManualInitialize
            ? new TOperator() : default);
    }

    /// <summary>
    /// src := Invoke(alpha)
    /// </summary>
    public static void Apply<TAction, TIn>(
        this VectorView src, TIn alpha, TAction? action = null)
        where TAction : struct, IUnaryOperator<TIn, double>
        where TIn : struct
    {
        Details.Apply_Impl
            (ref src.GetHeadRef(), alpha, src.Indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Apply_Impl<TAction, TIn>(ref double xHead,
             TIn alpha, SingleIndice indice, TAction action)
            where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        {
            if (indice.Length == 0)
                return;
            else if (indice.Stride > 1)
                Apply_Kernel
                    (ref xHead, alpha, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Apply_Kernel_Vector
                    (indice.Length, ref xHead, alpha, action);
            else
                Apply_Kernel_Unit
                    (indice.Length, ref xHead, alpha, action);
        }

        public static void Apply_Kernel_Vector<TAction, TIn>(
            int length, ref double xHead, TIn alpha, TAction action)
            where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        {
            int iterStride = 4;
            int i = 0;
            if (length >= iterStride * Vector<double>.Count)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                Vector<TIn> alphaVec = Vector.Create(alpha);
                var xVec0 = action.Invoke(in alphaVec);
                for (; i <= length - iterStride * Vector<double>.Count; i += iterStride * Vector<double>.Count)
                {
                    xVec0.StoreUnsafe(ref xHead);
                    xVec0.StoreUnsafe(ref xHead1);
                    xVec0.StoreUnsafe(ref xHead2);
                    xVec0.StoreUnsafe(ref xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterStride * Vector<double>.Count);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride * Vector<double>.Count);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride * Vector<double>.Count);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride * Vector<double>.Count);
                }
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    xVec0.StoreUnsafe(ref xHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                xHead = action.Invoke(alpha);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Apply_Kernel_Unit<TAction, TIn>(
            int length, ref double xHead, TIn alpha, TAction action)
            where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        {
            int iterSize = 4;
            int i = 0;
            var result = action.Invoke(alpha);
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    xHead = result;
                    xHead1 = result;
                    xHead2 = result;
                    xHead3 = result;
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                xHead = result;
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Apply_Kernel<TAction, TIn>(
            ref double xHead, TIn alpha, SingleIndice indice, TAction action)
                        where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        {
            int iterSize = 4;
            int iterStride = iterSize * indice.Stride;
            int i = 0;
            var result = action.Invoke(alpha);
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.Stride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.Stride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.Stride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    xHead = result;
                    xHead1 = result;
                    xHead2 = result;
                    xHead3 = result;
                    xHead = ref Unsafe.Add(ref xHead, iterStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                xHead = result;
                xHead = ref Unsafe.Add(ref xHead, indice.Stride);
            }
        }
    }

    /// <summary>
    /// src := Invoke(src)
    /// </summary>
    public static void Map<TAction>(this VectorView src, TAction? action = null)
                where TAction : struct, IUnaryOperator<double, double>
    {
        Details.Map_Impl(ref src.GetHeadRef(), src.Indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction>(
            ref double xHead, SingleIndice indice, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            if (indice.Length == 0)
                return;
            else if (indice.Stride > 1)
                Map_Kernel(ref xHead, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Map_Kernel_Vector(indice.Length, ref xHead, action);
            else
                Map_Kernel_Unit(indice.Length, ref xHead, action);
        }

        public static void Map_Kernel_Vector<TAction>(
            int length, ref double xHead, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            int iterStride = 4;
            int i = 0;
            if (length >= iterStride * Vector<double>.Count)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                for (; i <= length - iterStride * Vector<double>.Count; i += iterStride * Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> xVec1 = Vector.LoadUnsafe(ref xHead1);
                    Vector<double> xVec2 = Vector.LoadUnsafe(ref xHead2);
                    Vector<double> xVec3 = Vector.LoadUnsafe(ref xHead3);
                    xVec0 = action.Invoke(in xVec0);
                    xVec1 = action.Invoke(in xVec1);
                    xVec2 = action.Invoke(in xVec2);
                    xVec3 = action.Invoke(in xVec3);
                    xVec0.StoreUnsafe(ref xHead);
                    xVec1.StoreUnsafe(ref xHead1);
                    xVec2.StoreUnsafe(ref xHead2);
                    xVec3.StoreUnsafe(ref xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterStride * Vector<double>.Count);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride * Vector<double>.Count);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride * Vector<double>.Count);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride * Vector<double>.Count);
                }
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    xVec0 = action.Invoke(in xVec0);
                    xVec0.StoreUnsafe(ref xHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                xHead = action.Invoke(xHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Map_Kernel_Unit<TAction>(
            int length, ref double xHead, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            int iterSize = 4;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    xHead = action.Invoke(xHead);
                    xHead1 = action.Invoke(xHead1);
                    xHead2 = action.Invoke(xHead2);
                    xHead3 = action.Invoke(xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                xHead = action.Invoke(xHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Map_Kernel<TAction>(
            ref double xHead, SingleIndice indice, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            int iterSize = 4;
            int iterStride = iterSize * indice.Stride;
            int i = 0;
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.Stride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.Stride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.Stride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    xHead = action.Invoke(xHead);
                    xHead1 = action.Invoke(xHead1);
                    xHead2 = action.Invoke(xHead2);
                    xHead3 = action.Invoke(xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                xHead = action.Invoke(xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.Stride);
            }
        }
    }

    /// <summary>
    /// src := Invoke(src, alpha)
    /// </summary>
    public static void Map<TAction, TIn>(
        this VectorView src, TIn alpha, TAction? action = null)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        Details.Map_Impl(ref src.GetHeadRef(),
            alpha, src.Indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction, TIn>(
            ref double xHead, TIn alpha, SingleIndice indice, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            if (indice.Length == 0)
                return;
            else if (indice.Stride > 1)
                Map_Kernel
                    (ref xHead, alpha, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Map_Kernel_Vector
                    (indice.Length, ref xHead, alpha, action);
            else
                Map_Kernel_Unit
                    (indice.Length, ref xHead, alpha, action);
        }

        public static void Map_Kernel_Vector<TAction, TIn>(
            int length, ref double xHead, TIn alpha, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            int iterStride = 4;
            int i = 0;
            if (length >= iterStride * Vector<double>.Count)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                var alphaVec = Vector.Create(alpha);
                for (; i <= length - iterStride * Vector<double>.Count; i += iterStride * Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> xVec1 = Vector.LoadUnsafe(ref xHead1);
                    Vector<double> xVec2 = Vector.LoadUnsafe(ref xHead2);
                    Vector<double> xVec3 = Vector.LoadUnsafe(ref xHead3);
                    xVec0 = action.Invoke(in xVec0, in alphaVec);
                    xVec1 = action.Invoke(in xVec1, in alphaVec);
                    xVec2 = action.Invoke(in xVec2, in alphaVec);
                    xVec3 = action.Invoke(in xVec3, in alphaVec);
                    xVec0.StoreUnsafe(ref xHead);
                    xVec1.StoreUnsafe(ref xHead1);
                    xVec2.StoreUnsafe(ref xHead2);
                    xVec3.StoreUnsafe(ref xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterStride * Vector<double>.Count);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride * Vector<double>.Count);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride * Vector<double>.Count);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride * Vector<double>.Count);
                }
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    xVec0 = action.Invoke(in xVec0, in alphaVec);
                    xVec0.StoreUnsafe(ref xHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                xHead = action.Invoke(xHead, alpha);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Map_Kernel_Unit<TAction, TIn>(int length,
            ref double xHead, TIn alpha, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            int iterSize = 4;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    xHead = action.Invoke(xHead, alpha);
                    xHead1 = action.Invoke(xHead1, alpha);
                    xHead2 = action.Invoke(xHead2, alpha);
                    xHead3 = action.Invoke(xHead3, alpha);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                xHead = action.Invoke(xHead, alpha);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Map_Kernel<TAction, TIn>(ref double xHead,
            TIn alpha, SingleIndice indice, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            int iterSize = 4;
            int iterStride = iterSize * indice.Stride;
            int i = 0;
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.Stride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.Stride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.Stride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    xHead = action.Invoke(xHead, alpha);
                    xHead1 = action.Invoke(xHead1, alpha);
                    xHead2 = action.Invoke(xHead2, alpha);
                    xHead3 = action.Invoke(xHead3, alpha);
                    xHead = ref Unsafe.Add(ref xHead, iterStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                xHead = action.Invoke(xHead, alpha);
                xHead = ref Unsafe.Add(ref xHead, indice.Stride);
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src)
    /// </summary>
    public static void Map<TAction>(
        this VectorView src, VectorView dest, TAction? action = null)
        where TAction : struct, IUnaryOperator<double, double>
    {
        var indice = CheckIndice(src, dest);
        Details.Map_Impl(
            ref src.GetHeadRef(), ref dest.GetHeadRef(), indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction>(
            ref double xHead, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            if (indice.Length == 0)
                return;
            else if (indice.AStride > 1 || indice.BStride > 1)
                Map_Kernel
                    (ref xHead, ref yHead, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Map_Kernel_Vector
                    (indice.Length, ref xHead, ref yHead, action);
            else
                Map_Kernel_Unit
                    (indice.Length, ref xHead, ref yHead, action);
        }

        public static void Map_Kernel_Vector<TAction>(
            int length, ref double xHead, ref double yHead, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            int iterStride = 4;
            int iterSize = iterStride * Vector<double>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> xVec1 = Vector.LoadUnsafe(ref xHead1);
                    Vector<double> xVec2 = Vector.LoadUnsafe(ref xHead2);
                    Vector<double> xVec3 = Vector.LoadUnsafe(ref xHead3);
                    var yVec0 = action.Invoke(in xVec0);
                    var yVec1 = action.Invoke(in xVec1);
                    var yVec2 = action.Invoke(in xVec2);
                    var yVec3 = action.Invoke(in xVec3);
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
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> yVec0 = Vector.LoadUnsafe(ref yHead);
                    yVec0 = action.Invoke(in xVec0);
                    yVec0.StoreUnsafe(ref yHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                    yHead = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Map_Kernel_Unit<TAction>(
            int length, ref double xHead, ref double yHead, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            int iterSize = 4;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var yHead1 = ref Unsafe.Add(ref yHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead);
                    yHead1 = action.Invoke(xHead1);
                    yHead2 = action.Invoke(xHead2);
                    yHead3 = action.Invoke(xHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Map_Kernel<TAction>(
            ref double xHead, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            int i = 0;
            int iterSize = 4;
            int xIterStride = iterSize * indice.AStride;
            int yIterStride = iterSize * indice.BStride;
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.AStride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.AStride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.AStride * 3);
                ref var yHead1 = ref Unsafe.Add(ref yHead, indice.BStride);
                ref var yHead2 = ref Unsafe.Add(ref yHead, indice.BStride * 2);
                ref var yHead3 = ref Unsafe.Add(ref yHead, indice.BStride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead);
                    yHead1 = action.Invoke(xHead1);
                    yHead2 = action.Invoke(xHead2);
                    yHead3 = action.Invoke(xHead3);
                    xHead = ref Unsafe.Add(ref xHead, xIterStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, xIterStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, xIterStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, xIterStride);
                    yHead = ref Unsafe.Add(ref yHead, yIterStride);
                    yHead1 = ref Unsafe.Add(ref yHead1, yIterStride);
                    yHead2 = ref Unsafe.Add(ref yHead2, yIterStride);
                    yHead3 = ref Unsafe.Add(ref yHead3, yIterStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                yHead = action.Invoke(xHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src, alpha)
    /// </summary>
    public static void Map<TAction, TIn>(
        this VectorView src, TIn alpha, VectorView dest, TAction? action = null)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        var indice = CheckIndice(src, dest);
        Details.Map_Impl
            (ref src.GetHeadRef(),
            alpha, ref dest.GetHeadRef(), indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction, TIn>(
            ref double xHead, TIn alpha, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            if (indice.Length == 0)
                return;
            else if (indice.AStride > 1 || indice.BStride > 1)
                Map_Kernel
                    (ref xHead, alpha, ref yHead, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Map_Kernel_Vector
                    (indice.Length, ref xHead, alpha, ref yHead, action);
            else
                Map_Kernel_Unit
                    (indice.Length, ref xHead, alpha, ref yHead, action);
        }

        public static void Map_Kernel_Vector<TAction, TIn>(
            int length, ref double xHead, TIn alpha, ref double yHead, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            int iterStride = 4;
            int iterSize = iterStride * Vector<double>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 3);
                var alphaVec = Vector.Create(alpha);
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> xVec1 = Vector.LoadUnsafe(ref xHead1);
                    Vector<double> xVec2 = Vector.LoadUnsafe(ref xHead2);
                    Vector<double> xVec3 = Vector.LoadUnsafe(ref xHead3);
                    var yVec0 = action.Invoke(in xVec0, in alphaVec);
                    var yVec1 = action.Invoke(in xVec1, in alphaVec);
                    var yVec2 = action.Invoke(in xVec2, in alphaVec);
                    var yVec3 = action.Invoke(in xVec3, in alphaVec);
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
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> yVec0 = Vector.LoadUnsafe(ref yHead);
                    yVec0 = action.Invoke(in xVec0, in alphaVec);
                    yVec0.StoreUnsafe(ref yHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                    yHead = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead, alpha);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Map_Kernel_Unit<TAction, TIn>(
            int length, ref double xHead, TIn alpha, ref double yHead, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            int iterSize = 4;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var yHead1 = ref Unsafe.Add(ref yHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead, alpha);
                    yHead1 = action.Invoke(xHead1, alpha);
                    yHead2 = action.Invoke(xHead2, alpha);
                    yHead3 = action.Invoke(xHead3, alpha);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead, alpha);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Map_Kernel<TAction, TIn>(
            ref double xHead, TIn alpha, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            int i = 0;
            int iterSize = 4;
            int iterAStride = iterSize * indice.AStride;
            int iterBStride = iterSize * indice.BStride;
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.AStride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.AStride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.AStride * 3);
                ref var yHead1 = ref Unsafe.Add(ref yHead, indice.BStride);
                ref var yHead2 = ref Unsafe.Add(ref yHead, indice.BStride * 2);
                ref var yHead3 = ref Unsafe.Add(ref yHead, indice.BStride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead, alpha);
                    yHead1 = action.Invoke(xHead1, alpha);
                    yHead2 = action.Invoke(xHead2, alpha);
                    yHead3 = action.Invoke(xHead3, alpha);
                    xHead = ref Unsafe.Add(ref xHead, iterAStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterAStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterAStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterAStride);
                    yHead = ref Unsafe.Add(ref yHead, iterBStride);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterBStride);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterBStride);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterBStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                yHead = action.Invoke(xHead, alpha);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src, dest)
    /// </summary>
    public static void Combine<TAction>(
        this VectorView src, VectorView dest, TAction? action = null)
                where TAction : struct, IBinaryOperator<double, double, double>
    {
        var indice = CheckIndice(src, dest);
        Details.Combine_Impl(ref src.GetHeadRef(),
            ref dest.GetHeadRef(), indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Combine_Impl<TAction>(
            ref double xHead, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, IBinaryOperator<double, double, double>
        {
            if (indice.Length == 0)
                return;
            else if (indice.AStride > 1 || indice.BStride > 1)
                Combine_Kernel(ref xHead, ref yHead, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Combine_Kernel_Vector(indice.Length, ref xHead, ref yHead, action);
            else
                Combine_Kernel_Unit(indice.Length, ref xHead, ref yHead, action);
        }

        public static void Combine_Kernel_Vector<TAction>(
            int length, ref double xHead, ref double yHead, TAction action)
                        where TAction : struct, IBinaryOperator<double, double, double>
        {
            int iterStride = 4;
            int iterSize = iterStride * Vector<double>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> yVec0 = Vector.LoadUnsafe(ref yHead);
                    Vector<double> xVec1 = Vector.LoadUnsafe(ref xHead1);
                    Vector<double> yVec1 = Vector.LoadUnsafe(ref yHead1);
                    Vector<double> xVec2 = Vector.LoadUnsafe(ref xHead2);
                    Vector<double> yVec2 = Vector.LoadUnsafe(ref yHead2);
                    Vector<double> xVec3 = Vector.LoadUnsafe(ref xHead3);
                    Vector<double> yVec3 = Vector.LoadUnsafe(ref yHead3);
                    yVec0 = action.Invoke(in xVec0, in yVec0);
                    yVec1 = action.Invoke(in xVec1, in yVec1);
                    yVec2 = action.Invoke(in xVec2, in yVec2);
                    yVec3 = action.Invoke(in xVec3, in yVec3);
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
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> yVec0 = Vector.LoadUnsafe(ref yHead);
                    yVec0 = action.Invoke(in xVec0, in yVec0);
                    yVec0.StoreUnsafe(ref yHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                    yHead = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead, yHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Combine_Kernel_Unit<TAction>(
            int length, ref double xHead, ref double yHead, TAction action)
                        where TAction : struct, IBinaryOperator<double, double, double>
        {
            int iterSize = 4;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var yHead1 = ref Unsafe.Add(ref yHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead, yHead);
                    yHead1 = action.Invoke(xHead1, yHead1);
                    yHead2 = action.Invoke(xHead2, yHead2);
                    yHead3 = action.Invoke(xHead3, yHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead, yHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Combine_Kernel<TAction>(
            ref double xHead, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, IBinaryOperator<double, double, double>
        {
            int i = 0;
            int iterSize = 4;
            int iterAStride = iterSize * indice.AStride;
            int iterBStride = iterSize * indice.BStride;
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.AStride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.AStride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.AStride * 3);
                ref var yHead1 = ref Unsafe.Add(ref yHead, indice.BStride);
                ref var yHead2 = ref Unsafe.Add(ref yHead, indice.BStride * 2);
                ref var yHead3 = ref Unsafe.Add(ref yHead, indice.BStride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead, yHead);
                    yHead1 = action.Invoke(xHead1, yHead1);
                    yHead2 = action.Invoke(xHead2, yHead2);
                    yHead3 = action.Invoke(xHead3, yHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterAStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterAStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterAStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterAStride);
                    yHead = ref Unsafe.Add(ref yHead, iterBStride);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterBStride);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterBStride);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterBStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                yHead = action.Invoke(xHead, yHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src, alpha, dest)
    /// </summary>
    public static void Combine<TAction, TIn>(
        this VectorView src, TIn alpha, VectorView dest, TAction? action = null)
                where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
    {
        var indice = CheckIndice(src, dest);
        Details.Combine_Impl(ref src.GetHeadRef(),
            alpha, ref dest.GetHeadRef(), indice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Combine_Impl<TAction, TIn>(
            ref double xHead, TIn alpha, ref double yHead, DoubleIndice indice, TAction action)
                        where TAction : struct, ITernaryOperator<double, TIn, double, double>
            where TIn : struct
        {
            if (indice.Length == 0)
                return;
            else if (indice.AStride > 1 || indice.BStride > 1)
                Combine_Kernel(ref xHead, alpha, ref yHead, indice, action);
            else if (Vector.IsHardwareAccelerated && TAction.IsVectorizable)
                Combine_Kernel_Vector(indice.Length, ref xHead, alpha, ref yHead, action);
            else
                Combine_Kernel_Unit(indice.Length, ref xHead, alpha, ref yHead, action);
        }

        public static void Combine_Kernel_Vector<TAction, TIn>(
            int length, ref double xHead, TIn alpha, ref double yHead, TAction action)
                    where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
        {
            int iterStride = 4;
            int iterSize = iterStride * Vector<double>.Count;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 3);
                var alphaVec = Vector.Create(alpha);
                for (; i <= length - iterSize; i += iterSize)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> yVec0 = Vector.LoadUnsafe(ref yHead);
                    Vector<double> xVec1 = Vector.LoadUnsafe(ref xHead1);
                    Vector<double> yVec1 = Vector.LoadUnsafe(ref yHead1);
                    Vector<double> xVec2 = Vector.LoadUnsafe(ref xHead2);
                    Vector<double> yVec2 = Vector.LoadUnsafe(ref yHead2);
                    Vector<double> xVec3 = Vector.LoadUnsafe(ref xHead3);
                    Vector<double> yVec3 = Vector.LoadUnsafe(ref yHead3);
                    yVec0 = action.Invoke(in xVec0, in alphaVec, in yVec0);
                    yVec1 = action.Invoke(in xVec1, in alphaVec, in yVec1);
                    yVec2 = action.Invoke(in xVec2, in alphaVec, in yVec2);
                    yVec3 = action.Invoke(in xVec3, in alphaVec, in yVec3);
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
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    Vector<double> xVec0 = Vector.LoadUnsafe(ref xHead);
                    Vector<double> yVec0 = Vector.LoadUnsafe(ref yHead);
                    yVec0 = action.Invoke(in xVec0, in alphaVec, in yVec0);
                    yVec0.StoreUnsafe(ref yHead);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                    yHead = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead, alpha, yHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Combine_Kernel_Unit<TAction, TIn>(
            int length, ref double xHead, TIn alpha, ref double yHead, TAction action)
                        where TAction : struct, ITernaryOperator<double, TIn, double, double>
            where TIn : struct
        {
            int iterSize = 4;
            int i = 0;
            if (length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, 1);
                ref var yHead1 = ref Unsafe.Add(ref yHead, 1);
                ref var xHead2 = ref Unsafe.Add(ref xHead, 2);
                ref var yHead2 = ref Unsafe.Add(ref yHead, 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, 3);
                ref var yHead3 = ref Unsafe.Add(ref yHead, 3);
                for (; i <= length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead, alpha, yHead);
                    yHead1 = action.Invoke(xHead1, alpha, yHead1);
                    yHead2 = action.Invoke(xHead2, alpha, yHead2);
                    yHead3 = action.Invoke(xHead3, alpha, yHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterSize);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
                    yHead = ref Unsafe.Add(ref yHead, iterSize);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
                }
            }
            for (; i < length; i++)
            {
                yHead = action.Invoke(xHead, alpha, yHead);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void Combine_Kernel<TAction, TIn>(
            ref double xHead, TIn alpha, ref double yHead, DoubleIndice indice, TAction action)
            where TAction : struct, ITernaryOperator<double, TIn, double, double>
            where TIn : struct
        {
            int i = 0;
            int iterSize = 4;
            int iterAStride = iterSize * indice.AStride;
            int iterBStride = iterSize * indice.BStride;
            if (indice.Length >= iterSize)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.AStride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.AStride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.AStride * 3);
                ref var yHead1 = ref Unsafe.Add(ref yHead, indice.BStride);
                ref var yHead2 = ref Unsafe.Add(ref yHead, indice.BStride * 2);
                ref var yHead3 = ref Unsafe.Add(ref yHead, indice.BStride * 3);
                for (; i <= indice.Length - iterSize; i += iterSize)
                {
                    yHead = action.Invoke(xHead, alpha, yHead);
                    yHead1 = action.Invoke(xHead1, alpha, yHead1);
                    yHead2 = action.Invoke(xHead2, alpha, yHead2);
                    yHead3 = action.Invoke(xHead3, alpha, yHead3);
                    xHead = ref Unsafe.Add(ref xHead, iterAStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterAStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterAStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterAStride);
                    yHead = ref Unsafe.Add(ref yHead, iterBStride);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterBStride);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterBStride);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterBStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                yHead = action.Invoke(xHead, alpha, yHead);
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }

    internal static DoubleIndice CheckIndice(
        VectorView src, VectorView dest)
    {
        if (src.Length != dest.Length)
            throw new ArgumentException("Error: src and dest must have the same length.");
        return new(src.Length, src.Stride, dest.Stride);
    }

    internal static (DoubleIndice rowIndice, DoubleIndice colIndice) CheckIndice
        (MatrixView a, MatrixView b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException("Error: a and b must have the same dimensions.");
        return (new DoubleIndice(a.Rows, a.RowStride, b.RowStride),
                new DoubleIndice(a.Cols, a.ColStride, b.ColStride));
    }
}
