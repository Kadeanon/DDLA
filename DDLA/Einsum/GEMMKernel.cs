using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DDLA.Einsum;

public readonly struct GEMMKernel
{
    public int mc => 510;//2040;

    public int nc => 512;

    public int kc => 144;

    public int mr => 6;

    public int nr => 8;

    public int kr => 6;

    public bool preferCol => false;

    public void Kernel(Span<double> bufferA, Span<double> bufferB, Span<double> bufferC, int m, int n, int k)
    {
        ref var a = ref MemoryMarshal.GetReference(bufferA);
        ref var b = ref MemoryMarshal.GetReference(bufferB);
        ref var c = ref MemoryMarshal.GetReference(bufferC);
        Debug.Assert(n == nr);
        Debug.Assert(m == mr);
        MicroKernel(k, ref a, ref b, ref c);
    }


    public static unsafe void MicroKernel(int k,
        ref double a,
        ref double b,
        ref double c)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            Prefetch0(ref a);
            var c00Vec = Vector256<double>.Zero;
            var c04Vec = Vector256<double>.Zero;
            var c10Vec = Vector256<double>.Zero;
            var c14Vec = Vector256<double>.Zero;
            var c20Vec = Vector256<double>.Zero;
            var c24Vec = Vector256<double>.Zero;
            var c30Vec = Vector256<double>.Zero;
            var c34Vec = Vector256<double>.Zero;
            var c40Vec = Vector256<double>.Zero;
            var c44Vec = Vector256<double>.Zero;
            var c50Vec = Vector256<double>.Zero;
            var c54Vec = Vector256<double>.Zero;

            int k_iter = k / 4;
            int k_left = k % 4;

            for (int cycle = 0; cycle < k_iter; cycle++)
            {
                // iter 0
                Vector256<double> b0Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                Vector256<double> b4Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                Prefetch0(ref a, 8);
                Vector256<double> aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c00Vec = FMA(aVec, b0Vec, c00Vec);
                c04Vec = FMA(aVec, b4Vec, c04Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c10Vec = FMA(aVec, b0Vec, c10Vec);
                c14Vec = FMA(aVec, b4Vec, c14Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c20Vec = FMA(aVec, b0Vec, c20Vec);
                c24Vec = FMA(aVec, b4Vec, c24Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c30Vec = FMA(aVec, b0Vec, c30Vec);
                c34Vec = FMA(aVec, b4Vec, c34Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c40Vec = FMA(aVec, b0Vec, c40Vec);
                c44Vec = FMA(aVec, b4Vec, c44Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c50Vec = FMA(aVec, b0Vec, c50Vec);
                c54Vec = FMA(aVec, b4Vec, c54Vec);

                // iter 1
                b0Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                b4Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c00Vec = FMA(aVec, b0Vec, c00Vec);
                c04Vec = FMA(aVec, b4Vec, c04Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c10Vec = FMA(aVec, b0Vec, c10Vec);
                c14Vec = FMA(aVec, b4Vec, c14Vec);
                Prefetch0(ref a, 8);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c20Vec = FMA(aVec, b0Vec, c20Vec);
                c24Vec = FMA(aVec, b4Vec, c24Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c30Vec = FMA(aVec, b0Vec, c30Vec);
                c34Vec = FMA(aVec, b4Vec, c34Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c40Vec = FMA(aVec, b0Vec, c40Vec);
                c44Vec = FMA(aVec, b4Vec, c44Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c50Vec = FMA(aVec, b0Vec, c50Vec);
                c54Vec = FMA(aVec, b4Vec, c54Vec);

                // iter 2
                b0Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                b4Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c00Vec = FMA(aVec, b0Vec, c00Vec);
                c04Vec = FMA(aVec, b4Vec, c04Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c10Vec = FMA(aVec, b0Vec, c10Vec);
                c14Vec = FMA(aVec, b4Vec, c14Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c20Vec = FMA(aVec, b0Vec, c20Vec);
                c24Vec = FMA(aVec, b4Vec, c24Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c30Vec = FMA(aVec, b0Vec, c30Vec);
                c34Vec = FMA(aVec, b4Vec, c34Vec);
                Prefetch0(ref a, 8);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c40Vec = FMA(aVec, b0Vec, c40Vec);
                c44Vec = FMA(aVec, b4Vec, c44Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c50Vec = FMA(aVec, b0Vec, c50Vec);
                c54Vec = FMA(aVec, b4Vec, c54Vec);

                // iter 3
                b0Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                b4Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c00Vec = FMA(aVec, b0Vec, c00Vec);
                c04Vec = FMA(aVec, b4Vec, c04Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c10Vec = FMA(aVec, b0Vec, c10Vec);
                c14Vec = FMA(aVec, b4Vec, c14Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c20Vec = FMA(aVec, b0Vec, c20Vec);
                c24Vec = FMA(aVec, b4Vec, c24Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c30Vec = FMA(aVec, b0Vec, c30Vec);
                c34Vec = FMA(aVec, b4Vec, c34Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c40Vec = FMA(aVec, b0Vec, c40Vec);
                c44Vec = FMA(aVec, b4Vec, c44Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c50Vec = FMA(aVec, b0Vec, c50Vec);
                c54Vec = FMA(aVec, b4Vec, c54Vec);
            }
            for (int cycle = 1; cycle < k_left - 1; cycle++)
            {
                Prefetch0(ref a, 8 * cycle);
            }
            for (int cycle = 0; cycle < k_left; cycle++)
            {
                Vector256<double> b0Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                Vector256<double> b4Vec = Vector256.LoadUnsafe(ref b);
                b = ref Unsafe.Add(ref b, 4);
                Vector256<double> aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c00Vec = FMA(aVec, b0Vec, c00Vec);
                c04Vec = FMA(aVec, b4Vec, c04Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c10Vec = FMA(aVec, b0Vec, c10Vec);
                c14Vec = FMA(aVec, b4Vec, c14Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c20Vec = FMA(aVec, b0Vec, c20Vec);
                c24Vec = FMA(aVec, b4Vec, c24Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c30Vec = FMA(aVec, b0Vec, c30Vec);
                c34Vec = FMA(aVec, b4Vec, c34Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c40Vec = FMA(aVec, b0Vec, c40Vec);
                c44Vec = FMA(aVec, b4Vec, c44Vec);
                aVec = Vector256.Create(a);
                a = ref Unsafe.Add(ref a, 1);
                c50Vec = FMA(aVec, b0Vec, c50Vec);
                c54Vec = FMA(aVec, b4Vec, c54Vec);
            }

            c00Vec.StoreUnsafe(ref c);
            c04Vec.StoreUnsafe(ref Unsafe.Add(ref c, 4));
            c10Vec.StoreUnsafe(ref Unsafe.Add(ref c, 8));
            c14Vec.StoreUnsafe(ref Unsafe.Add(ref c, 12));
            c20Vec.StoreUnsafe(ref Unsafe.Add(ref c, 16));
            c24Vec.StoreUnsafe(ref Unsafe.Add(ref c, 20));
            c30Vec.StoreUnsafe(ref Unsafe.Add(ref c, 24));
            c34Vec.StoreUnsafe(ref Unsafe.Add(ref c, 28));
            c40Vec.StoreUnsafe(ref Unsafe.Add(ref c, 32));
            c44Vec.StoreUnsafe(ref Unsafe.Add(ref c, 36));
            c50Vec.StoreUnsafe(ref Unsafe.Add(ref c, 40));
            c54Vec.StoreUnsafe(ref Unsafe.Add(ref c, 44));
        }
        else
        {
            throw new NotImplementedException(
                "Vector128 or non-SIMD kernel is not supported on this platform. " +
                "Please use a different kernel implementation or ensure your platform supports Vector256 operations.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<double> FMA(Vector256<double> a,
        Vector256<double> b, Vector256<double> c)
    {
        if (Fma.IsSupported)
        {
            return Fma.MultiplyAdd(a, b, c);
        }
        else
        {
            return Vector256.Add(Vector256.Multiply(a, b), c);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Prefetch0(ref double @ref, int offset)
    {
        if (Sse.IsSupported)
            Sse.Prefetch0(
                Unsafe.AsPointer(ref Unsafe.Add(ref @ref, offset)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Prefetch0(ref double @ref)
    {
        if (Sse.IsSupported)
            Sse.Prefetch0(
                Unsafe.AsPointer(ref @ref));
    }
}
