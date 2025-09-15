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
        MatrixView A, TIn alpha, TAction action)
                where TAction : struct, IUnaryOperator<TIn, double>
        where TIn : struct
    {
        var rowIndice = A.RowIndice;
        var colIndice = A.ColIndice;
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Apply_Impl(ref A.GetHeadRef(),
            alpha, A.RowIndice, A.ColIndice, action);
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
    /// A := Invoke(A)
    /// </summary>
    public static void Map<TAction>
        (MatrixView A, TAction action)
                where TAction : struct, IUnaryOperator<double, double>
    {
        var rowIndice = A.RowIndice;
        var colIndice = A.ColIndice;
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl(ref A.GetHeadRef(),
            A.RowIndice, A.ColIndice, action);
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
    /// A := Invoke(A, alpha)
    /// </summary>
    public static void Map<TAction, TIn>(
        MatrixView A, TIn alpha, TAction action)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        var rowIndice = A.RowIndice;
        var colIndice = A.ColIndice;
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl(ref A.GetHeadRef(),
            alpha, A.RowIndice, A.ColIndice, action);
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
    /// B := Invoke(A)
    /// </summary>
    public static void Map<TAction>(
        MatrixView A, MatrixView B, TAction action)
                where TAction : struct, IUnaryOperator<double, double>
    {
        var (rowIndice, colIndice) = CheckIndice(A, B);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl(ref A.GetHeadRef(),
            ref B.GetHeadRef(), rowIndice, colIndice, action);
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
    /// B := Invoke(A, alpha)
    /// </summary>
    public static void Map<TAction, TIn>(
        MatrixView A, TIn alpha, MatrixView B, 
        TAction action)
                where TAction : struct, IBinaryOperator<double, TIn, double>
        where TIn : struct
    {
        var (rowIndice, colIndice) = CheckIndice(A, B);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Map_Impl
            (ref A.GetHeadRef(), alpha, ref B.GetHeadRef(), 
            rowIndice, colIndice, action);
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
    /// B := Invoke(A, B)
    /// </summary>
    public static void Combine<TAction>(
        MatrixView A, MatrixView B, TAction action)
                where TAction : struct, IBinaryOperator<double, double, double>
    {
        var (rowIndice, colIndice) = CheckIndice(A, B);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Combine_Impl(ref A.GetHeadRef(),
            ref B.GetHeadRef(), rowIndice, colIndice, action);
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
    /// B := Invoke(A, alpha, B)
    /// </summary>
    public static void Combine<TAction, TIn>(
        MatrixView A, TIn alpha, MatrixView B, TAction action)
                where TAction : struct, ITernaryOperator<double, TIn, double, double>
        where TIn : struct
    {
        var (rowIndice, colIndice) = CheckIndice(A, B);
        if (rowIndice.Length == 0 || colIndice.Length == 0)
            return;
        Details.Combine_Impl(ref A.GetHeadRef(),
            alpha, ref B.GetHeadRef(), rowIndice, colIndice, action);
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
