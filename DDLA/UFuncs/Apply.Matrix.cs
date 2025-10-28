using DDLA.Core;
using DDLA.Misc;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs;

public static partial class UFunc
{

    /// <summary>
    /// x := Invoke(alpha)
    /// </summary>
    public static void Apply<TAction, TIn>(
        this MatrixView src, TIn alpha, TAction? action = null)
                where TAction : struct, IUnaryOperator<TIn, double>
        where TIn : struct
    {
        var rowIndice = src.RowIndice;
        var colIndice = src.ColIndice;
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Apply_Impl(ref src.GetHeadRef(),
            alpha, src.RowIndice, src.ColIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Apply_Impl<TAction, TIn>
            (ref double aHead, TIn alpha, 
            SingleIndice rowIndice, SingleIndice colIndice, 
            TAction action)
            where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        {
            if (rowIndice.Stride < colIndice.Stride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Apply_Impl
                    (ref aHead, alpha, colIndice, action);
            else if (colIndice.Length == 1)
                Apply_Impl
                    (ref aHead, alpha, rowIndice, action);
            else if (colIndice.Stride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Apply_Kernel
                        (ref aHead, alpha, colIndice, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.Stride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Apply_Kernel_Vector
                        (colIndice.Length, ref aHead, alpha, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.Stride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Apply_Kernel_Unit
                        (colIndice.Length, ref aHead, alpha, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.Stride);
                }
            }
        }
    }

    /// <summary>
    /// src := Invoke(src)
    /// </summary>
    public static void Map<TAction>
        (this MatrixView src, TAction? action = null)
                where TAction : struct, IUnaryOperator<double, double>
    {
        var rowIndice = src.RowIndice;
        var colIndice = src.ColIndice;
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl(ref src.GetHeadRef(),
            src.RowIndice, src.ColIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction>(
            ref double aHead, 
            SingleIndice rowIndice, SingleIndice colIndice,
            TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            if (rowIndice.Stride < colIndice.Stride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Map_Impl(ref aHead, colIndice, action);
            else if (colIndice.Length == 1)
                Map_Impl(ref aHead, rowIndice, action);
            else if (colIndice.Stride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel(ref aHead, colIndice, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.Stride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Vector
                        (colIndice.Length, ref aHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.Stride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Unit(colIndice.Length, ref aHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.Stride);
                }
            }
        }
    }

    /// <summary>
    /// src := Invoke(src, alpha)
    /// </summary>
    public static void Map<TAction, TIn>(
        this MatrixView src, TIn alpha, TAction? action = null)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        var rowIndice = src.RowIndice;
        var colIndice = src.ColIndice;
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl(ref src.GetHeadRef(),
            alpha, src.RowIndice, src.ColIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction, TIn>(
            ref double xHead, TIn alpha, 
            SingleIndice rowIndice, SingleIndice colIndice, 
            TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            if (rowIndice.Stride < colIndice.Stride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Map_Impl
                    (ref xHead, alpha, colIndice, action);
            else if (colIndice.Length == 1)
                Map_Impl
                    (ref xHead, alpha, rowIndice, action);
            else if (colIndice.Stride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel
                        (ref xHead, alpha, colIndice, action);
                    xHead = ref Unsafe.Add(ref xHead, rowIndice.Stride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Vector
                        (colIndice.Length, ref xHead, alpha, action);
                    xHead = ref Unsafe.Add(ref xHead, rowIndice.Stride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Unit
                        (colIndice.Length, ref xHead, alpha, action);
                    xHead = ref Unsafe.Add(ref xHead, rowIndice.Stride);
                }
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src)
    /// </summary>
    public static void Map<TAction>(
        this MatrixView src, MatrixView dest, TAction? action = null)
                where TAction : struct, IUnaryOperator<double, double>
    {
        var (rowIndice, colIndice) = CheckIndice(src, dest);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl(ref src.GetHeadRef(),
            ref dest.GetHeadRef(), rowIndice, colIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction>(
            ref double aHead, ref double bHead, 
            DoubleIndice rowIndice, DoubleIndice colIndice, 
            TAction action)
                        where TAction : struct, IUnaryOperator<double, double>
        {
            if (rowIndice.BStride < colIndice.BStride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Map_Impl
                    (ref aHead, ref bHead, colIndice, action);
            else if (colIndice.Length == 1)
                Map_Impl
                    (ref aHead, ref bHead, rowIndice, action);
            else if (colIndice.AStride > 1 || colIndice.BStride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel
                        (ref aHead, ref bHead, colIndice, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Vector
                        (colIndice.Length, ref aHead, ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Unit
                        (colIndice.Length, ref aHead, ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src, alpha)
    /// </summary>
    public static void Map<TAction, TIn>(
        this MatrixView src, TIn alpha, MatrixView dest,
        TAction? action = null)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        var (rowIndice, colIndice) = CheckIndice(src, dest);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl
            (ref src.GetHeadRef(), alpha, ref dest.GetHeadRef(), 
            rowIndice, colIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Map_Impl<TAction, TIn>(
            ref double aHead, TIn alpha, ref double bHead, 
            DoubleIndice rowIndice, DoubleIndice colIndice, 
            TAction action)
                        where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        {
            if (rowIndice.BStride < colIndice.BStride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Map_Impl
                    (ref aHead, alpha, ref bHead, colIndice, action);
            else if (colIndice.Length == 1)
                Map_Impl
                    (ref aHead, alpha, ref bHead, rowIndice, action);
            else if (colIndice.AStride > 1 || colIndice.BStride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel(ref aHead,
                        alpha, ref bHead, colIndice, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Vector(colIndice.Length,
                        ref aHead, alpha, ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Map_Kernel_Unit(colIndice.Length, ref aHead, alpha,
                        ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src, dest)
    /// </summary>
    public static void Combine<TAction>(
        this MatrixView src, MatrixView dest, TAction? action = null)
                where TAction : struct, IBinaryOperator<double, double, double>
    {
        var (rowIndice, colIndice) = CheckIndice(src, dest);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Combine_Impl(ref src.GetHeadRef(),
            ref dest.GetHeadRef(), rowIndice, colIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Combine_Impl<TAction>(
            ref double aHead, ref double bHead, 
            DoubleIndice rowIndice, DoubleIndice colIndice,
            TAction action)
                        where TAction : struct, IBinaryOperator<double, double, double>
        {
            if (rowIndice.BStride < colIndice.BStride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Combine_Impl(ref aHead, ref bHead, colIndice, action);
            else if (colIndice.Length == 1)
                Combine_Impl(ref aHead, ref bHead, rowIndice, action);
            else if (colIndice.AStride > 1 || colIndice.BStride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Combine_Kernel(ref aHead, ref bHead,
                        colIndice, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Combine_Kernel_Vector(colIndice.Length,
                        ref aHead, ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Combine_Kernel_Unit(colIndice.Length,
                        ref aHead, ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
        }
    }

    /// <summary>
    /// dest := Invoke(src, alpha, dest)
    /// </summary>
    public static void Combine<TAction, TIn>(
        this MatrixView src, TIn alpha, MatrixView dest, TAction? action = null)
                where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
    {
        var (rowIndice, colIndice) = CheckIndice(src, dest);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Combine_Impl(ref src.GetHeadRef(),
            alpha, ref dest.GetHeadRef(), rowIndice, colIndice, action.OrDefault());
    }

    public static partial class Details
    {
        public static void Combine_Impl<TAction, TIn>(
            ref double aHead, TIn alpha, ref double bHead,
            DoubleIndice rowIndice, DoubleIndice colIndice, 
            TAction action)
                where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
        {
            if (rowIndice.BStride < colIndice.BStride)
            {
                (rowIndice, colIndice) =
                    (colIndice, rowIndice);
            }
            if (rowIndice.Length == 1)
                Combine_Impl(ref aHead,
                    alpha, ref bHead, colIndice, action);
            else if (colIndice.Length == 1)
                Combine_Impl(ref aHead,
                    alpha, ref bHead, rowIndice, action);
            else if (colIndice.AStride > 1 || colIndice.BStride > 1)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Combine_Kernel(ref aHead,
                        alpha, ref bHead, colIndice, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else if (System.Numerics.Vector.IsHardwareAccelerated &&
                TAction.IsVectorizable)
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Combine_Kernel_Vector(colIndice.Length,
                        ref aHead, alpha, ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
            else
            {
                for (int i = 0; i < rowIndice.Length; i++)
                {
                    Combine_Kernel_Unit(colIndice.Length, ref aHead, alpha,
                        ref bHead, action);
                    aHead = ref Unsafe.Add(ref aHead, rowIndice.AStride);
                    bHead = ref Unsafe.Add(ref bHead, rowIndice.BStride);
                }
            }
        }
    }

}
