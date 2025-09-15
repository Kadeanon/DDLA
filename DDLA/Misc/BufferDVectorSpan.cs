using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace DDLA.Misc;

public unsafe ref struct BufferDVectorSpan : IDisposable
{
    public ref double bufferHead;
    public ref double oldHead;
    public double[]? managedBuffer;
    public int oldLength;
    public int oldStride;
    private bool bufferAllocated;
    private readonly bool shouldCopyBack;

    private readonly bool Managed => oldLength <= int.MaxValue / 2;

    public BufferDVectorSpan(ref double head, int length, int stride, bool shouldCopyBack = false, bool forceCopy = false)
    {
        bufferHead = ref head;
        oldHead = ref head;
        oldStride = stride;
        oldLength = length;
        if (length > 0 && stride > 1 || forceCopy)
        {
            this.shouldCopyBack = shouldCopyBack;
            if (Managed)
            {
                int bufferLength = length;
                managedBuffer = ArrayPool<double>.Shared.Rent(bufferLength);
                managedBuffer.AsSpan(0, bufferLength).Clear();
                bufferHead = ref managedBuffer[0];
            }
            else
            {
                bufferHead = ref Unsafe.AsRef<double>(
                NativeMemory.AlignedAlloc(
                (nuint)(length * sizeof(double)),
                (nuint)Unsafe.SizeOf<Vector256<double>>()));
            }
            bufferAllocated = true;
            DoubleIndice indice = new(length, stride, 1);
            UFunc.Details.Map_Impl<IdentityOperator<double>>
            (ref head, ref bufferHead, indice, default);
        }
    }

    public BufferDVectorSpan(ref double head, int length, int stride, double scale, bool shouldCopyBack = false)
    {
        bufferHead = ref head;
        oldHead = ref head;
        oldStride = stride;
        oldLength = length;
        this.shouldCopyBack = shouldCopyBack;
        if (length == 0)
        {
            shouldCopyBack = false;
            bufferAllocated = false;
            return;
        }

        if (Managed)
        {
            int bufferLength = length;
            managedBuffer = ArrayPool<double>.Shared.Rent(length);
            managedBuffer.AsSpan(0, bufferLength).Clear();
            bufferHead = ref managedBuffer[0];
        }
        else
        {
            bufferHead = ref Unsafe.AsRef<double>(
            NativeMemory.AlignedAlloc(
            (nuint)(length * sizeof(double)),
            (nuint)Unsafe.SizeOf<Vector256<double>>()));
        }
        bufferAllocated = true;
        DoubleIndice indice = new(length, stride, 1);
        if (scale != 0.0)
            UFunc.Details.Map_Impl
                <MultiplyOperator<double>, double>(
                ref head, scale, ref bufferHead, indice, default);
    }

    public void Dispose()
    {
        if (bufferAllocated)
        {
            if (shouldCopyBack)
            {
                DoubleIndice indice = new(oldLength, 1, oldStride);
                UFunc.Details.Map_Impl<IdentityOperator<double>>
                    (ref bufferHead, ref oldHead, indice, default);
            }
            if (Managed)
            {
                int bufferLength = oldLength;
                managedBuffer.AsSpan(0, bufferLength).Clear();
                ArrayPool<double>.Shared.Return(managedBuffer!);
            }
            else
                NativeMemory.AlignedFree(Unsafe.AsPointer(ref bufferHead));
            bufferAllocated = false;
        }
    }
}
