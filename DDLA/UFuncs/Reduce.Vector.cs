using System.Numerics;
using System.Runtime.CompilerServices;
using DDLA.Core;
using DDLA.Misc;
using DDLA.UFuncs.Operators;
using Vector = System.Numerics.Vector;

namespace DDLA.UFuncs;

public static partial class UFunc
{
    /// <summary>
    /// return reduce(trans(x))
    /// </summary>
    public static TOut Reduce<TTranspose, TReducer, TMid, TOut>(
        VectorView x)
        where TTranspose : struct, IUnaryOperator<double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
    {
        var reducer = new TReducer();
        Details.Reduce_Impl<
            TTranspose, TReducer, TMid, TOut>(
            ref x.GetHeadRef(), x.Indice, new(), ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(trans(x))
    /// </summary>
    public static TOut Reduce<TTranspose, TReducer, TMid, TOut>(
        VectorView x, TTranspose transposer, ref TReducer reducer)
        where TTranspose : struct, IUnaryOperator<double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
    {
        Details.Reduce_Impl<
            TTranspose, TReducer, TMid, TOut>(
            ref x.GetHeadRef(), x.Indice, transposer, ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(trans(x))
    /// </summary>
    public static double Reduce<TTranspose, TReducer>(
        VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        where TReducer : struct, IAggregationOperator<double, double>
    {
        var reducer = new TReducer();
        Details.Reduce_Impl<
            TTranspose, TReducer, double, double>(
            ref x.GetHeadRef(), x.Indice,
            new(), ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(trans(x))
    /// </summary>
    public static double Reduce<TTranspose, TReducer>(
        VectorView x, TTranspose transposer, ref TReducer reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        where TReducer : struct, IAggregationOperator<double, double>
    {
        Details.Reduce_Impl<
            TTranspose, TReducer, double, double>(
            ref x.GetHeadRef(), x.Indice,
            transposer, ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(x)
    /// </summary>
    public static double Reduce<TReducer>(
        VectorView x)
        where TReducer : struct, IAggregationOperator<double, double>
    {
        var reducer = new TReducer();
        Details.Reduce_Impl<
            IdentityOperator<double>, TReducer, double, double>(
            ref x.GetHeadRef(), x.Indice,
            new(), ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(x)
    /// </summary>
    public static double Reduce<TReducer>(
        VectorView x, ref TReducer reducer)
        where TReducer : struct, IAggregationOperator<double, double>
    {
        Details.Reduce_Impl<
            IdentityOperator<double>, TReducer, double, double>(
            ref x.GetHeadRef(), x.Indice,
            new(), ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return max(trans(x))
    /// </summary>
    public static double Max<TTranspose>(VectorView x,
        ref MaxAggregationOperator<double> reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MaxAggregationOperator<double>,
            double, double>(x, new(), ref reducer);

    /// <summary>
    /// return max(trans(x))
    /// </summary>
    public static double Max<TTranspose>(VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MaxAggregationOperator<double>,
            double, double>(x);

    /// <summary>
    /// return maxnumber(trans(x))
    /// </summary>
    public static double MaxNumber<TTranspose>(VectorView x,
        ref MaxNumberAggregationOperator<double> reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MaxNumberAggregationOperator<double>,
            double, double>(x, new(), ref reducer);

    /// <summary>
    /// return maxnumber(trans(x))
    /// </summary>
    public static double MaxNumber<TTranspose>(VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MaxNumberAggregationOperator<double>,
            double, double>(x);

    /// <summary>
    /// return min(trans(x))
    /// </summary>
    public static double Min<TTranspose>(VectorView x,
        ref MinAggregationOperator<double> reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MinAggregationOperator<double>,
            double, double>(x, new(), ref reducer);

    /// <summary>
    /// return min(trans(x))
    /// </summary>
    public static double Min<TTranspose>(VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MinAggregationOperator<double>,
            double, double>(x);

    /// <summary>
    /// return minnumber(trans(x))
    /// </summary>
    public static double MinNumber<TTranspose>(VectorView x,
        ref MinNumberAggregationOperator<double> reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MinNumberAggregationOperator<double>,
            double, double>(x, new(), ref reducer);

    /// <summary>
    /// return minnumber(trans(x))
    /// </summary>
    public static double MinNumber<TTranspose>(VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            MinNumberAggregationOperator<double>,
            double, double>(x);

    /// <summary>
    /// return sum(trans(x))
    /// </summary>
    public static double Sum<TTranspose>(VectorView x,
        ref SumOperator<double> reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            SumOperator<double>,
            double, double>(x, new(), ref reducer);

    /// <summary>
    /// return sum(trans(x))
    /// </summary>
    public static double Sum<TTranspose>(VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            SumOperator<double>,
            double, double>(x);

    /// <summary>
    /// return product(trans(x))
    /// </summary>
    public static double Product<TTranspose>(VectorView x,
        ref ProductOperator<double> reducer)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            ProductOperator<double>,
            double, double>(x, new(), ref reducer);

    /// <summary>
    /// return product(trans(x))
    /// </summary>
    public static double Product<TTranspose>(VectorView x)
        where TTranspose : struct, IUnaryOperator<double, double>
        => Reduce<TTranspose,
            ProductOperator<double>,
            double, double>(x);

    public static partial class Details
    {
        public static void Reduce_Impl<TTranspose, TReducer, TMid, TOut>(
            ref double xHead, SingleIndice indice, TTranspose transposer, ref TReducer reducer)
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TTranspose : struct, IUnaryOperator<double, TMid>
        where TOut : struct
        where TMid : struct
        {
            if (indice.Length == 0)
                return;
            else if (indice.Stride == 1 && Vector.IsHardwareAccelerated
                && Vector<TOut>.IsSupported && TReducer.IsVectorizable)
                Reduce_Kernel_Vector<TTranspose, TReducer, TMid, TOut>(indice.Length,
                    ref xHead, transposer, ref reducer);
            else
                Reduce_Kernel<TTranspose, TReducer, TMid, TOut>
                    (ref xHead, indice, transposer, ref reducer);
        }

        public static void Reduce_Kernel_Vector<TTranspose, TReducer, TMid, TOut>(
            int length, ref double xHead, TTranspose transposer, ref TReducer reducer)
                where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TTranspose : struct, IUnaryOperator<double, TMid>
        where TOut : struct
        where TMid : struct
        {
            const int iterUnrolling = 4;
            int i = 0;
            if (length >= iterUnrolling * Vector<double>.Count)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                var valVec = Vector.Create(TReducer.Seed);
                for (; i <= length -
                    iterUnrolling * Vector<double>.Count;
                    i += iterUnrolling * Vector<double>.Count)
                {
                    var xVec0 = Vector.LoadUnsafe(ref xHead);
                    var xVec1 = Vector.LoadUnsafe(ref xHead1);
                    var xVec2 = Vector.LoadUnsafe(ref xHead2);
                    var xVec3 = Vector.LoadUnsafe(ref xHead3);
                    var transVec0 = transposer.Invoke(in xVec0);
                    var transVec1 = transposer.Invoke(in xVec1);
                    var transVec2 = transposer.Invoke(in xVec2);
                    var transVec3 = transposer.Invoke(in xVec3);
                    valVec = reducer.Invoke(in valVec, in transVec0);
                    valVec = reducer.Invoke(in valVec, in transVec1);
                    valVec = reducer.Invoke(in valVec, in transVec2);
                    valVec = reducer.Invoke(in valVec, in transVec3);
                    xHead = ref Unsafe.Add(ref xHead, iterUnrolling * Vector<double>.Count);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterUnrolling * Vector<double>.Count);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterUnrolling * Vector<double>.Count);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterUnrolling * Vector<double>.Count);
                }
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    var xVec0 = Vector.LoadUnsafe(ref xHead);
                    var transVec0 = transposer.Invoke(in xVec0);
                    valVec = reducer.Invoke(in valVec, in transVec0);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                }
                reducer.Combine(in valVec);
            }
            for (; i < length; i++)
            {
                var trans = transposer.Invoke(xHead);
                reducer.Combine(trans);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void Reduce_Kernel<TTranspose, TReducer, TMid, TOut>(
            ref double xHead, SingleIndice indice, TTranspose transposer, ref TReducer reducer)
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TTranspose : struct, IUnaryOperator<double, TMid>
        where TOut : struct
        where TMid : struct
        {
            int iterUnrolling = 4;
            int iterStride = iterUnrolling * indice.Stride;
            int i = 0;
            if (indice.Length >= iterUnrolling)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.Stride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.Stride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.Stride * 3);
                for (; i <= indice.Length - iterUnrolling; i += iterUnrolling)
                {
                    reducer.Combine(transposer.Invoke(xHead));
                    reducer.Combine(transposer.Invoke(xHead1));
                    reducer.Combine(transposer.Invoke(xHead2));
                    reducer.Combine(transposer.Invoke(xHead3));
                    xHead = ref Unsafe.Add(ref xHead, iterStride);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride);
                }
            }
            for (; i < indice.Length; i++)
            {
                reducer.Combine(transposer.Invoke(xHead));
                xHead = ref Unsafe.Add(ref xHead, indice.Stride);
            }
        }
    }

    /// <summary>
    /// return reduce(zip(x, y))
    /// </summary>
    public static TOut ZipReduce<TZipper, TReducer, TMid, TOut>(
        VectorView x, VectorView y)
        where TZipper : struct, IBinaryOperator<double, double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
    {
        var reducer = new TReducer();
        var indice = CheckIndice(x, y);
        Details.ZipReduce_Impl<
            TZipper, TReducer, TMid, TOut>(
            ref x.GetHeadRef(), ref y.GetHeadRef(), indice, new(), ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(zip(x, y))
    /// </summary>
    public static TOut ZipReduce<TZipper, TReducer, TMid, TOut>(
        VectorView x, VectorView y, TZipper zipper, ref TReducer reducer)
        where TZipper : struct, IBinaryOperator<double, double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
    {
        var indice = CheckIndice(x, y);
        Details.ZipReduce_Impl<
            TZipper, TReducer, TMid, TOut>(
            ref x.GetHeadRef(), ref y.GetHeadRef(), indice, zipper, ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(zip(x, y))
    /// </summary>
    public static double ZipReduce<TZipper, TReducer>(
        VectorView x, VectorView y)
        where TZipper : struct, IBinaryOperator<double, double, double>
        where TReducer : struct, IAggregationOperator<double, double>
    {
        var reducer = new TReducer();
        var indice = CheckIndice(x, y);
        Details.ZipReduce_Impl<
            TZipper, TReducer, double, double>(
            ref x.GetHeadRef(), ref y.GetHeadRef(), indice, new(), ref reducer);
        return reducer.Result;
    }

    /// <summary>
    /// return reduce(zip(x, y))
    /// </summary>
    public static double ZipReduce<TZipper, TReducer>(
        VectorView x, VectorView y, TZipper zipper, ref TReducer reducer)
        where TZipper : struct, IBinaryOperator<double, double, double>
        where TReducer : struct, IAggregationOperator<double, double>
    {
        var indice = CheckIndice(x, y);
        Details.ZipReduce_Impl<
            TZipper, TReducer, double, double>(
            ref x.GetHeadRef(), ref y.GetHeadRef(), indice, zipper, ref reducer);
        return reducer.Result;
    }

    public static partial class Details
    {
        public static void ZipReduce_Impl<TZipper, TReducer, TMid, TOut>(
            ref double xHead, ref double yHead, DoubleIndice indice, 
            TZipper zipper, ref TReducer reducer)
        where TZipper : struct, IBinaryOperator<double, double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
        {
            if (indice.Length == 0)
                return;
            else if (indice.AStride == 1 && indice.BStride == 1 &&
                Vector.IsHardwareAccelerated &&
                Vector<TOut>.IsSupported &&
                TReducer.IsVectorizable)
                ZipReduce_Kernel_Vector<TZipper, TReducer, TMid, TOut>(
                    indice.Length, ref xHead, ref yHead, zipper, ref reducer);
            else
                ZipReduce_Kernel<TZipper, TReducer, TMid, TOut>(
                    ref xHead, ref yHead, indice, zipper, ref reducer);
        }

        public static void ZipReduce_Kernel_Vector<
            TZipper, TReducer, TMid, TOut>(
            int length, ref double xHead, ref double yHead, TZipper zipper, ref TReducer reducer)
        where TZipper : struct, IBinaryOperator<double, double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
        {
            var iterStride = 4;
            var i = 0;
            if (length >= iterStride * Vector<double>.Count)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                ref var xHead2 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, Vector<double>.Count * 3);
                ref var yHead1 = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                ref var yHead2 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 2);
                ref var yHead3 = ref Unsafe.Add(ref yHead, Vector<double>.Count * 3);
                var valVec = Vector.Create(TReducer.Seed);
                for (; i <= length - iterStride * Vector<double>.Count; i += iterStride * Vector<double>.Count)
                {
                    var xVec0 = Vector.LoadUnsafe(ref xHead);
                    var xVec1 = Vector.LoadUnsafe(ref xHead1);
                    var xVec2 = Vector.LoadUnsafe(ref xHead2);
                    var xVec3 = Vector.LoadUnsafe(ref xHead3);
                    var yVec0 = Vector.LoadUnsafe(ref yHead);
                    var yVec1 = Vector.LoadUnsafe(ref yHead1);
                    var yVec2 = Vector.LoadUnsafe(ref yHead2);
                    var yVec3 = Vector.LoadUnsafe(ref yHead3);
                    var transVec0 = zipper.Invoke(in xVec0, in yVec0);
                    var transVec1 = zipper.Invoke(in xVec1, in yVec1);
                    var transVec2 = zipper.Invoke(in xVec2, in yVec2);
                    var transVec3 = zipper.Invoke(in xVec3, in yVec3);
                    valVec = reducer.Invoke(in valVec, in transVec0);
                    valVec = reducer.Invoke(in valVec, in transVec1);
                    valVec = reducer.Invoke(in valVec, in transVec2);
                    valVec = reducer.Invoke(in valVec, in transVec3);
                    xHead = ref Unsafe.Add(ref xHead, iterStride * Vector<double>.Count);
                    xHead1 = ref Unsafe.Add(ref xHead1, iterStride * Vector<double>.Count);
                    xHead2 = ref Unsafe.Add(ref xHead2, iterStride * Vector<double>.Count);
                    xHead3 = ref Unsafe.Add(ref xHead3, iterStride * Vector<double>.Count);
                    yHead = ref Unsafe.Add(ref yHead, iterStride * Vector<double>.Count);
                    yHead1 = ref Unsafe.Add(ref yHead1, iterStride * Vector<double>.Count);
                    yHead2 = ref Unsafe.Add(ref yHead2, iterStride * Vector<double>.Count);
                    yHead3 = ref Unsafe.Add(ref yHead3, iterStride * Vector<double>.Count);
                }
                for (; i <= length - Vector<double>.Count; i += Vector<double>.Count)
                {
                    var xVec0 = Vector.LoadUnsafe(ref xHead);
                    var yVec0 = Vector.LoadUnsafe(ref yHead);
                    var transVec0 = zipper.Invoke(in xVec0, in yVec0);
                    valVec = reducer.Invoke(in valVec, in transVec0);
                    xHead = ref Unsafe.Add(ref xHead, Vector<double>.Count);
                    yHead = ref Unsafe.Add(ref yHead, Vector<double>.Count);
                }
                reducer.Combine(in valVec);
            }
            for (; i < length; i++)
            {
                var trans = zipper.Invoke(xHead, yHead);
                reducer.Combine(trans);
                xHead = ref Unsafe.Add(ref xHead, 1);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void ZipReduce_Kernel<
            TZipper, TReducer, TMid, TOut>(
            ref double xHead, ref double yHead, DoubleIndice indice, TZipper zipper, ref TReducer reducer)
        where TZipper : struct, IBinaryOperator<double, double, TMid>
        where TReducer : struct, IAggregationOperator<TMid, TOut>
        where TOut : struct
        where TMid : struct
        {
            int iterUnrolling = 4;
            int iterAStride = iterUnrolling * indice.AStride;
            int iterBStride = iterUnrolling * indice.BStride;
            int i = 0;
            if (indice.Length >= iterUnrolling)
            {
                ref var xHead1 = ref Unsafe.Add(ref xHead, indice.AStride);
                ref var xHead2 = ref Unsafe.Add(ref xHead, indice.AStride * 2);
                ref var xHead3 = ref Unsafe.Add(ref xHead, indice.AStride * 3);
                ref var yHead1 = ref Unsafe.Add(ref yHead, indice.BStride);
                ref var yHead2 = ref Unsafe.Add(ref yHead, indice.BStride * 2);
                ref var yHead3 = ref Unsafe.Add(ref yHead, indice.BStride * 3);
                for (; i <= indice.Length - iterUnrolling; i += iterUnrolling)
                {
                    reducer.Combine(zipper.Invoke(xHead, yHead));
                    reducer.Combine(zipper.Invoke(xHead1, yHead1));
                    reducer.Combine(zipper.Invoke(xHead2, yHead2));
                    reducer.Combine(zipper.Invoke(xHead3, yHead3));
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
                reducer.Combine(zipper.Invoke(xHead, yHead));
                xHead = ref Unsafe.Add(ref xHead, indice.AStride);
                yHead = ref Unsafe.Add(ref yHead, indice.BStride);
            }
        }
    }
}