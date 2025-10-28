using DDLA.Core;
using DDLA.Misc;
using DDLA.Utilities;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs;

public static partial class UFunc
{
    public static MArray Apply<TAction, TIn>
        (this MArray array, TIn alpha, TAction? action = null)
        where TAction : struct, IUnaryOperator<TIn, double>
        where TIn : struct
    {
        if (array.Rank == 0)
            return array;
        
        var instance = action.OrDefault();

        if (array.Rank == 1)
        {
            Details.Apply_Impl(ref array.GetHeadRef(),
                alpha, new SingleIndice(array.Lengths[0], array.Strides[0]),
                instance);
            return array;
        }
        else
        {
            Parallel.For(0, array.Lengths[0], (int index) =>
            {
                ref var head = ref Unsafe.Add(ref array.GetHeadRef(),
                    index * array.Strides[0]);
                Span<SingleIndice> indices =
                stackalloc SingleIndice[array.Rank - 1];
                for (int i = 0; i < array.Rank - 1; i++)
                {
                    indices[i] = new SingleIndice(array.Lengths[i + 1],
                        array.Strides[i + 1]);
                }
                Details.Apply_Impl(ref head, alpha, indices, instance);
            });
            return array;
        }
    }

    public partial class Details
    {
        public static void Apply_Impl<TAction, TIn>(
            ref double head, TIn alpha, ReadOnlySpan<SingleIndice> indices, TAction action)
            where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        {
            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Apply_Impl(ref head,
                    alpha, indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Apply_Impl(ref head,
                    alpha, indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Apply_Impl(ref head, alpha, subIndices, action);
                    head = ref Unsafe.Add(ref head, firstIndice.Stride);
                }
            }
        }
    }

    public static MArray Map<TAction>
        (this MArray array, TAction? action = null)
        where TAction : struct, IUnaryOperator<double, double>
    {
        if (array.Rank == 0)
            return array;

        var instance = action.OrDefault();

        if (array.Rank == 1)
        {
            Details.Map_Impl(ref array.GetHeadRef(),
                new SingleIndice(array.Lengths[0], array.Strides[0]), instance);
            return array;
        }
        else
        {
            Parallel.For(0, array.Lengths[0], index =>
            {
                var subArray = array.SliceFirstDim(index);
                ref var head = ref subArray.Data[subArray.Offset];
                Span<SingleIndice> indices =
                stackalloc SingleIndice[subArray.Rank];
                for (int i = 0; i < subArray.Rank; i++)
                {
                    indices[i] = new SingleIndice(subArray.Lengths[i],
                        subArray.Strides[i]);
                }
                Details.Map_Impl(ref head, indices, instance);
            });
            return array;
        }
    }

    public partial class Details
    {
        public static void Map_Impl<TAction>(
        ref double head, ReadOnlySpan<SingleIndice> indices, TAction action)
                where TAction : struct, IUnaryOperator<double, double>
        {
            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Map_Impl(ref head, indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Map_Impl(ref head,
                    indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Map_Impl(ref head, subIndices, action);
                    head = ref Unsafe.Add(ref head, firstIndice.Stride);
                }
            }
        }
    }

    public static MArray Map<TAction, TIn>(
         this MArray array, TIn alpha, TAction? action = null)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        if (array.Rank == 0)
            return array;

        var instance = action.OrDefault();

        if (array.Rank == 1)
        {
            Details.Map_Impl(ref array.GetHeadRef(),
                alpha, new SingleIndice(array.Lengths[0], array.Strides[0]), instance);
            return array;
        }
        else
        {
            Parallel.For(0, array.Lengths[0], index =>
            {
                var subArray = array.SliceFirstDim(index);
                ref var head = ref subArray.Data[subArray.Offset];
                Span<SingleIndice> indices =
                stackalloc SingleIndice[subArray.Rank];
                for (int i = 0; i < subArray.Rank; i++)
                {
                    indices[i] = new SingleIndice(subArray.Lengths[i],
                        subArray.Strides[i]);
                }
                Details.Map_Impl(ref head, alpha, indices, instance);
            });
            return array;
        }
    }

    public partial class Details
    {
        public static void Map_Impl<TAction, TIn>(
            ref double head, TIn alpha, 
            ReadOnlySpan<SingleIndice> indices, TAction action)
                    where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {

            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Map_Impl(
                    ref head, alpha, indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Map_Impl(ref head,
                    alpha, indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Map_Impl(ref head, alpha, subIndices, action);
                    head = ref Unsafe.Add(ref head, firstIndice.Stride);
                }
            }
        }
    }

    public static void Map<TAction>(
        this MArray src, MArray dest, TAction? action = null)
        where TAction : struct, IUnaryOperator<double, double>
    {
        ThrowUtils.ThrowIfArrayNotMatch(src, dest);
        if (src.Rank == 0)
            return;

        var instance = action.OrDefault();
        DoubleIndice[] indices = new DoubleIndice[src.Rank];
        for (int i = 0; i < src.Rank; i++)
        {
            indices[i] = new DoubleIndice(src.Lengths[i],
                src.Strides[i], dest.Strides[i]);
        }
        Array.Sort(indices, (x, y) => y.BStride.CompareTo(x.BStride));
        var srcStrideMinIndex = -1;
        var srcStrideMin = int.MaxValue;
        for (int i = 0; i < src.Rank; i++)
        {
            if (indices[i].AStride < srcStrideMin)
            {
                srcStrideMin = indices[i].AStride;
                srcStrideMinIndex = i;
            }
        }
        if (srcStrideMinIndex != 0 && srcStrideMinIndex != src.Rank)
        {
            (indices[^1], indices[srcStrideMinIndex]) =
                (indices[srcStrideMinIndex], indices[^1]);
        }
        if (indices.Length == 1)
        {
            ref var srcHead = ref src.Data[src.Offset];
            ref var destHead = ref dest.Data[dest.Offset];
            Details.Map_Impl(ref srcHead,
                 ref destHead, indices[0], instance);
            return;
        }
        else
        {
            var head = indices[0];
            indices = indices[1..];
            Parallel.For(0, head.Length, index =>
            {
                ref var srcRef = ref src.Data[
                    src.Offset + index * head.AStride];
                ref var destRef = ref dest.Data[
                    dest.Offset + index * head.BStride];
                Details.Map_Impl(ref srcRef, ref destRef, indices, instance);
            });
            return;
        }
    }

    public partial class Details
    {
        public static void Map_Impl<TAction>(
        ref double srcHead, ref double destHead, 
        ReadOnlySpan<DoubleIndice> indices, TAction action)
                where TAction : struct, IUnaryOperator<double, double>
        {
            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Map_Impl(ref srcHead,
                     ref destHead, indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Map_Impl(ref srcHead,
                    ref destHead, indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Map_Impl(ref srcHead, ref destHead,
                        subIndices, action);
                    srcHead = ref Unsafe.Add(ref srcHead, firstIndice.AStride);
                    destHead = ref Unsafe.Add(ref destHead, firstIndice.BStride);
                }
            }
        }
    }

    public static void Map<TAction, TIn>(
         this MArray src, TIn alpha, MArray dest, TAction? action = null)
        where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        ThrowUtils.ThrowIfArrayNotMatch(src, dest);
        if (src.Rank == 0)
            return;

        var instance = action.OrDefault();
        DoubleIndice[] indices = new DoubleIndice[src.Rank];
        for (int i = 0; i < src.Rank; i++)
        {
            indices[i] = new DoubleIndice(src.Lengths[i],
                src.Strides[i], dest.Strides[i]);
        }
        Array.Sort(indices, (x, y) => y.BStride.CompareTo(x.BStride));
        var srcStrideMinIndex = -1;
        var srcStrideMin = int.MaxValue;
        for (int i = 0; i < src.Rank; i++)
        {
            if (indices[i].AStride < srcStrideMin)
            {
                srcStrideMin = indices[i].AStride;
                srcStrideMinIndex = i;
            }
        }
        if (srcStrideMinIndex != 0 && srcStrideMinIndex != src.Rank)
        {
            (indices[^1], indices[srcStrideMinIndex]) =
                (indices[srcStrideMinIndex], indices[^1]);
        }
        if (indices.Length == 1)
        {
            ref var srcHead = ref src.Data[src.Offset];
            ref var destHead = ref dest.Data[dest.Offset];
            Details.Map_Impl
                (ref srcHead, alpha, ref destHead, indices[0], instance);
            return;
        }
        else
        {
            var head = indices[0];
            indices = indices[1..];
            Parallel.For(0, head.Length, index =>
            {
                ref var srcRef = ref src.Data[
                    src.Offset + index * head.AStride];
                ref var destRef = ref dest.Data[
                    dest.Offset + index * head.BStride];
                Details.Map_Impl
                (ref srcRef, alpha, ref destRef, indices, instance);
            });
            return;
        }
    }

    public partial class Details
    {
        public static void Map_Impl<TAction, TIn>(
        ref double srcHead, TIn alpha, ref double destHead,
        ReadOnlySpan<DoubleIndice> indices, TAction action)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
        {
            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Map_Impl(ref srcHead,
                    alpha, ref destHead, indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Map_Impl(ref srcHead,
                    alpha, ref destHead, indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Map_Impl(ref srcHead, alpha,
                        ref destHead, subIndices, action);
                    srcHead = ref Unsafe.Add(ref srcHead, firstIndice.AStride);
                    destHead = ref Unsafe.Add(ref destHead, firstIndice.BStride);
                }
            }
        }
    }

    public static void Combine<TAction>(
         this MArray src, MArray dest, TAction? action = null)
        where TAction : struct, IBinaryOperator<double, double, double>
    {
        ThrowUtils.ThrowIfArrayNotMatch(src, dest);
        if (src.Rank == 0)
            return;

        var instance = action.OrDefault();
        DoubleIndice[] indices = new DoubleIndice[src.Rank];
        for (int i = 0; i < src.Rank; i++)
        {
            indices[i] = new DoubleIndice(src.Lengths[i],
                src.Strides[i], dest.Strides[i]);
        }
        Array.Sort(indices, (x, y) => y.BStride.CompareTo(x.BStride));
        var srcStrideMinIndex = -1;
        var srcStrideMin = int.MaxValue;
        for (int i = 0; i < src.Rank; i++)
        {
            if (indices[i].AStride < srcStrideMin)
            {
                srcStrideMin = indices[i].AStride;
                srcStrideMinIndex = i;
            }
        }
        if (srcStrideMinIndex != 0 && srcStrideMinIndex != src.Rank)
        {
            (indices[^1], indices[srcStrideMinIndex]) =
                (indices[srcStrideMinIndex], indices[^1]);
        }
        if (indices.Length == 1)
        {
            ref var srcHead = ref src.Data[src.Offset];
            ref var destHead = ref dest.Data[dest.Offset];
            Details.Combine_Impl
                (ref srcHead, ref destHead, indices[0], instance);
            return;
        }
        else
        {
            var head = indices[0];
            indices = indices[1..];
            Parallel.For(0, head.Length, index =>
            {
                ref var srcRef = ref src.Data[
                    src.Offset + index * head.AStride];
                ref var destRef = ref dest.Data[
                    dest.Offset + index * head.BStride];
                Details.Combine_Impl(ref srcRef, ref destRef, indices, instance);
            });
            return;
        }
    }

    public partial class Details
    {
        public static void Combine_Impl<TAction>(
        ref double srcHead, ref double destHead,
        ReadOnlySpan<DoubleIndice> indices, TAction action)
                where TAction : struct, IBinaryOperator<double, double, double>
        {
            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Combine_Impl(ref srcHead, ref destHead,
                    indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Combine_Impl(ref srcHead, ref destHead,
                    indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Combine_Impl(ref srcHead, ref destHead, subIndices, action);
                    srcHead = ref Unsafe.Add(ref srcHead, firstIndice.AStride);
                    destHead = ref Unsafe.Add(ref destHead, firstIndice.BStride);
                }
            }
        }
    }

    public static void Combine<TAction, TIn>(
         this MArray src, TIn alpha, MArray dest, TAction? action = null)
                where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
    {
        ThrowUtils.ThrowIfArrayNotMatch(src, dest);
        if (src.Rank == 0)
            return;

        var instance = action.OrDefault();
        DoubleIndice[] indices = new DoubleIndice[src.Rank];
        for (int i = 0; i < src.Rank; i++)
        {
            indices[i] = new DoubleIndice(src.Lengths[i],
                src.Strides[i], dest.Strides[i]);
        }
        Array.Sort(indices, (x, y) => y.BStride.CompareTo(x.BStride));
        var srcStrideMinIndex = -1;
        var srcStrideMin = int.MaxValue;
        for (int i = 0; i < src.Rank; i++)
        {
            if (indices[i].AStride < srcStrideMin)
            {
                srcStrideMin = indices[i].AStride;
                srcStrideMinIndex = i;
            }
        }
        if (srcStrideMinIndex != 0 && srcStrideMinIndex != src.Rank)
        {
            (indices[^1], indices[srcStrideMinIndex]) =
                (indices[srcStrideMinIndex], indices[^1]);
        }
        if (indices.Length == 1)
        {
            ref var srcHead = ref src.Data[src.Offset];
            ref var destHead = ref dest.Data[dest.Offset];
            Details.Combine_Impl
                (ref srcHead, alpha, ref destHead, indices[0], instance);
            return;
        }
        else
        {
            var head = indices[0];
            indices = indices[1..];
            Parallel.For(0, head.Length, index =>
            {
                ref var srcRef = ref src.Data[
                    src.Offset + index * head.AStride];
                ref var destRef = ref dest.Data[
                    dest.Offset + index * head.BStride];
                Details.Combine_Impl(
                    ref srcRef, alpha, ref destRef, indices, instance);
            });
            return;
        }
    }

    public partial class Details
    {
        public static void Combine_Impl<TAction, TIn>(
        ref double srcHead, TIn alpha, ref double destHead,
        ReadOnlySpan<DoubleIndice> indices, TAction action)
                where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
        {
            if (indices.Length == 0)
                return;
            else if (indices.Length == 1)
            {
                Combine_Impl(ref srcHead,
                    alpha, ref destHead, indices[0], action);
            }
            else if (indices.Length == 2)
            {
                Combine_Impl(ref srcHead,
                    alpha, ref destHead, indices[0], indices[1], action);
            }
            else
            {
                var firstIndice = indices[0];
                var subIndices = indices[1..];
                for (var i = 0; i < firstIndice.Length; i++)
                {
                    Combine_Impl(ref srcHead, alpha,
                        ref destHead, subIndices, action);
                    srcHead = ref Unsafe.Add(ref srcHead, firstIndice.AStride);
                    destHead = ref Unsafe.Add(ref destHead, firstIndice.BStride);
                }
            }
        }
    }
}
