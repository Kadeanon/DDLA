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
using DDLA.UFuncs;
using System.Runtime.Intrinsics.X86;

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

    public static void Shift(scalar alpha, in vector x)
        => x.Map<AddOperator<scalar>, scalar>(alpha);

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

        Managed.BlasProvider.Details.RotInner(in x, in y, giv.c, giv.s);
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
                Managed.BlasProvider.Details.RotInner(y, z, giv2.c, giv2.s);
        }
        else if (giv2.c == 1 && giv2.s == 0)
        {
            Managed.BlasProvider.Details.RotInner(x, y, giv1.c, giv1.s);
        }
        else
        {
            Managed.BlasProvider.Details.Rot2Inner(x, y, z, giv1.c, giv1.s, giv2.c, giv2.s);
        }
    }
}
