using DDLA.Utilities;
using System.Collections;

namespace DDLA.Core;

public partial class MArray : IEnumerable<double>
{

    IEnumerator IEnumerable.GetEnumerator()
        => GetEnumerator();

    IEnumerator<double> IEnumerable<double>.GetEnumerator()
        => GetEnumerator();

    public MArraySegements AsSegements()
        => new(this);

    public MArrayEnumerator GetEnumerator()
        => new(this);

    public struct MArraySegements
    {
        private MArray Array { get; }
        public int Index { get; private set; }
        public int Length { get; }
        public int Batch { get; }
        public int Stride { get; }
        public readonly Span<int> StateSpan => state;

        public ArraySegment<int> state;

        public ArraySegment<int> dimLengths;

        public ArraySegment<int> dimStrides;

        public VectorView Current { get; set; }

        public int Step { get; set; }

        public MArraySegements(MArray array)
        {
            if (array.Rank == 0)
            {
                // For empty array, we can set it to the invalid state.
                Array = array;
                ReadOnlySpan<int> lengths = array.Lengths;
                Batch = lengths[0];
                Length = 0;
                Stride = 1;
                state = [];
                dimLengths = [];
                dimStrides = [];
                Index = 0;
                Current = VectorView.Empty;
            }
            else if (array.Rank == 1)
            {
                // For 1D array, we can use the VectorSpan directly.
                Array = array;
                ReadOnlySpan<int> lengths = array.Lengths;
                Batch = lengths[0];
                Length = 1;
                Stride = array.Strides[0];
                state = [];
                dimLengths = [];
                dimStrides = [];
                Index = -1;
            }
            else
            {
                Array = array;
                ReadOnlySpan<int> lengths = array.Lengths;
                Batch = lengths[^1];
                Length = lengths[..^1].Product();
                Stride = array.Strides[^1];
                state = new int[Array.Rank - 1];
                var rank = Array.Rank;
                dimLengths = new(Array.Metadata, 0, rank - 1);
                dimStrides = new(Array.Metadata, rank, rank - 1);
                Index = -1;
            }
        }

        public bool MoveNext()
        {
            if (Length <= 1)
            {
                if (Index >= 0)
                    return false;
                else
                {
                    Index = 0;
                    Current = new(Array.Data, Array.Offset,
                        Batch, Stride);
                    return true;
                }
            }
            int dimLength = dimLengths.Count;
            var stateSpan = StateSpan;
            if (Index > -1)
            {
                Step = IntUtils.IncrementIndexLeft(dimLength - 1, stateSpan, dimLengths);
            }
            Index++;
            var stridesSpan = Array.Strides[..^1];
            int index = Array.Offset + IntUtils.Dot(stateSpan, stridesSpan);
            Current = new(Array.Data, index, Batch, Stride);
            return Index < Length;
        }
    }

    public struct MArrayEnumerator(MArray array)
        : IEnumerator<double>
    {
        MArraySegements sequences = new(array);

        int elementIndex = -1;

        public readonly ref double Current =>
            ref sequences.Current[elementIndex];

        double IEnumerator<double>.Current => Current;

        object IEnumerator.Current => Current;

        public void Dispose() { }

        public bool MoveNext()
        {
            elementIndex = (elementIndex + 1) % sequences.Batch;
            if (elementIndex == 0)
                return sequences.MoveNext();
            return true;
        }

        public void Reset()
        {
            throw new NotImplementedException();
        }
    }
}
