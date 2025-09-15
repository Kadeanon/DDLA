using SingleIndices = System.Collections.Generic.Dictionary<char, DDLA.Misc.SingleIndice>;
using DoubleIndices = System.Collections.Generic.Dictionary<char, DDLA.Misc.DoubleIndice>;
using TripleIndices = System.Collections.Generic.Dictionary<char, DDLA.Misc.TripleIndice>;
using DDLA.Misc;
using DDLA.Utilities;
using DDLA.Core;

namespace DDLA.Utilities;

public static class IndiceUtils
{

    /// <summary>
    /// Creates a dictionary of indices from the tensor's rank and the provided symbol.
    /// Removes duplicate characters and combines their strides.
    /// </summary>
    /// <exception cref="ArgumentException"></exception>
    public static SingleIndices Diagonal(this MArray tensor, ReadOnlySpan<char> symbol)
    {
        if (tensor.Rank != symbol.Length)
        {
            throw new ArgumentException("The rank of the tensor must match the length of the symbol.");
        }

        SingleIndices result = new(symbol.Length);
        var lengths = tensor.Lengths;
        var strides = tensor.Strides;
        for (int i = 0; i < symbol.Length; i++)
        {
            char c = symbol[i];
            int length = lengths[i];
            int stride = strides[i];
            if (result.TryGetValue(c, out var indice))
            {
                if (indice.Length != length)
                {
                    throw new ArgumentException($"The character '{c}' already exists with different lengths.");
                }
                result[c] = new(length, indice.Stride + stride);
                continue;
            }
            else
            {
                result[c] = new(length, stride);
            }
        }
        return result;
    }

    public static void Divide(SingleIndices aIndex, SingleIndices bIndex,
        out DoubleIndices abIndex)
    {
        abIndex = new();
        foreach (var aItem in aIndex)
        {
            char aChar = aItem.Key;
            var aIndice = aItem.Value;
            if (bIndex.Remove(aChar, out var bIndice))
            {
                if (bIndice.Length != aIndice.Length)
                {
                    throw new ArgumentException($"The character '{aChar}' exists in both indices with different lengths.");
                }
                abIndex[aChar] = new DoubleIndice(aIndice.Length, aIndice.Stride, bIndice.Stride);
            }
        }
        foreach (var abItem in abIndex)
        {
            aIndex.Remove(abItem.Key);
        }
    }

    public static void Divide(SingleIndices aIndex, SingleIndices bIndex,
        SingleIndices cIndex, out DoubleIndices abIndex, out DoubleIndices acIndex,
        out DoubleIndices bcIndex, out TripleIndices abcIndex)
    {
        Divide(aIndex, bIndex, out abIndex);
        acIndex = new();
        bcIndex = new();
        abcIndex = new();
        foreach (var cItem in cIndex)
        {
            char cChar = cItem.Key;
            var cIndice = cItem.Value;
            if (abIndex.Remove(cChar, out var abIndice))
            {
                if (abIndice.Length != cIndice.Length)
                {
                    throw new ArgumentException($"The character '{cChar}' exists in both indices with different lengths.");
                }
                abcIndex[cChar] = new(abIndice.Length, abIndice.AStride, abIndice.BStride, cIndice.Stride);
            }
            else if (aIndex.Remove(cChar, out var aIndice))
            {
                if (aIndice.Length != cIndice.Length)
                {
                    throw new ArgumentException($"The character '{cChar}' exists in both indices with different lengths.");
                }
                acIndex[cChar] = new DoubleIndice(aIndice.Length, aIndice.Stride, cIndice.Stride);
            }
            else if (bIndex.Remove(cChar, out var bIndice))
            {
                if (bIndice.Length != cIndice.Length)
                {
                    throw new ArgumentException($"The character '{cChar}' exists in both indices with different lengths.");
                }
                bcIndex[cChar] = new DoubleIndice(bIndice.Length, bIndice.Stride, cIndice.Stride);
            }
        }

        foreach (var acItem in acIndex)
        {
            cIndex.Remove(acItem.Key);
        }

        foreach (var bcItem in bcIndex)
        {
            cIndex.Remove(bcItem.Key);
        }

        foreach (var abcItem in abcIndex)
        {
            cIndex.Remove(abcItem.Key);
        }
    }

    public static void Fold(this SingleIndices index)
    {
        if (index.Count < 2)
            return;

        Dictionary<int, char> dummy = new(index.Count);

        foreach ((var c, var indice) in index)
            dummy[indice.Length * indice.Stride] = c;

        bool feld;
        do
            feld = Fold_Inner(index, dummy);
        while (feld);
    }

    private static bool Fold_Inner(SingleIndices index, Dictionary<int, char> dummy)
    {
        char label = default;
        SingleIndice indiceToFold = default;
        int stride = default;
        char labelToBeFold = default;
        bool found = false;
        foreach (var indice in index)
        {
            stride = indice.Value.Stride;
            if (dummy.TryGetValue(stride, out labelToBeFold)
                && labelToBeFold != label)
            {
                found = true;
                (label, indiceToFold) = indice;
                break;
            }
        }
        if (!found)
            return false;

        while (found)
        {
            index.Remove(labelToBeFold, out var another);
            dummy.Remove(stride);
            stride = another.Stride;
            indiceToFold = new(indiceToFold.Length * another.Length, stride);

            found = false;
            foreach (var anoIndice in index)
            {
                if (dummy.TryGetValue(anoIndice.Value.Length, out labelToBeFold)
                && labelToBeFold != label)
                {
                    found = true;
                    break;
                }
            }
        }
        index[label] = indiceToFold;
        return true;
    }

    public static void Fold(this DoubleIndices index)
    {
        if (index.Count < 2)
            return;

        Dictionary<(int, int), char> dummy = new(index.Count);

        foreach ((var c, var indice) in index)
            dummy[(indice.Length * indice.AStride,
                indice.Length * indice.BStride)] = c;

        bool feld;
        do
            feld = Fold_Inner(index, dummy);
        while (feld);
    }

    private static bool Fold_Inner(DoubleIndices index, Dictionary<(int, int), char> dummy)
    {
        char label = default;
        DoubleIndice indiceToFold = default;
        (int, int) stride = default;
        char labelToBeFold = default;
        bool found = false;
        foreach (var indice in index)
        {
            stride = (indice.Value.AStride, indice.Value.BStride);
            if (dummy.TryGetValue(stride, out labelToBeFold)
                && labelToBeFold != label)
            {
                found = true;
                (label, indiceToFold) = indice;
                break;
            }
        }
        if (!found)
            return false;

        while (found)
        {
            index.Remove(labelToBeFold, out var another);
            dummy.Remove(stride);
            stride = (another.AStride, another.BStride);
            indiceToFold = new(indiceToFold.Length * another.Length, another.AStride, another.BStride);

            found = false;
            foreach (var anoIndice in index)
            {
                if (dummy.TryGetValue(stride, out labelToBeFold)
                && labelToBeFold != label)
                {
                    found = true;
                    break;
                }
            }
        }
        index[label] = indiceToFold;
        return true;
    }

    public static void Fold(this TripleIndices index)
    {
        if (index.Count < 2)
            return;

        Dictionary<(int, int, int), char> dummy = new(index.Count);

        foreach ((var c, var indice) in index)
            dummy[(indice.Length * indice.AStride,
                indice.Length * indice.BStride,
                indice.Length * indice.CStride)] = c;

        bool feld;
        do
            feld = Fold_Inner(index, dummy);
        while (feld);
    }

    private static bool Fold_Inner(TripleIndices index,
        Dictionary<(int, int, int), char> dummy)
    {
        char label = default;
        TripleIndice indiceToFold = default;
        (int, int, int) stride = default;
        char labelToBeFold = default;
        bool found = false;
        foreach (var indice in index)
        {
            stride = (indice.Value.AStride, indice.Value.BStride, indice.Value.CStride);
            if (dummy.TryGetValue(stride, out labelToBeFold)
                && labelToBeFold != label)
            {
                found = true;
                (label, indiceToFold) = indice;
                break;
            }
        }
        if (!found)
            return false;

        while (found)
        {
            index.Remove(labelToBeFold, out var another);
            dummy.Remove(stride);
            stride = (another.AStride, another.BStride, another.CStride);
            indiceToFold = new(indiceToFold.Length * another.Length,
                another.AStride, another.BStride, another.CStride);

            found = false;
            foreach (var anoIndice in index)
            {
                if (dummy.TryGetValue(stride, out labelToBeFold)
                && labelToBeFold != label)
                {
                    found = true;
                    break;
                }
            }
        }
        index[label] = indiceToFold;
        return true;
    }

    public static void Sort(this Span<SingleIndice> indices)
    {
        indices.Sort((a, b) =>
        {
            if (a.Stride != b.Stride)
                return -a.Stride.CompareTo(b.Stride);
            return -a.Length.CompareTo(b.Length);
        });
    }

    public static int TotalLength(this Span<SingleIndice> indices)
    {
        if (indices.Length == 0)
            return 0;
        int result = 1;
        foreach (var indice in indices)
        {
            result *= indice.Length;
        }
        return result;
    }

    public static int TotalLength(this Span<DoubleIndice> indices)
    {
        if (indices.Length == 0)
            return 0;
        int result = 1;
        foreach (var indice in indices)
        {
            result *= indice.Length;
        }
        return result;
    }

    public static int Align(this int total, int block)
        => (total + block - 1) / block * block;
}
