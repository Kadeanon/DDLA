using DDLA.Core;
using DDLA.Misc;

namespace DDLA.Einsum.EinsumTree;

internal class EinsumState
{
    public char[] symbolTable;
    public Dictionary<char, int> symbolToIndex;
    public int[] dimLengths;
    public int dimCount;
    public int inputCount;
    public string target;

    public int[,] strides;
    public bool[,] used;
    public string[] tensorNames;
    public int[] tensorSize;
    public MArray?[] tensors;
    public EinsumNode?[] nodes;

    public int size;
    public int cost;
    public int left;
    public int right;
    public string expression;
    private EinsumState? child;
    private readonly EinsumState? parent;

    public int TotalCost
    {
        get
        {
            int cost = this.cost;
            var current = child;
            while (current != null)
            {
                cost += current.cost;
                current = current.child;
            }
            return cost;
        }
    }

    public EinsumState(string expression,
        params ReadOnlySpan<MArray> tensors)
    {
        HashSet<char> symbols = [];
        var parts = expression.Split("->",
            StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
        {
            throw new ArgumentException("Invalid einsum expression format.");
        }
        target = parts[1];
        tensorNames = parts[0].Split(',',
            StringSplitOptions.TrimEntries);
        inputCount = tensorNames.Length;
        if (inputCount == 0)
        {
            throw new ArgumentException("No tensors provided in the einsum expression.");
        }
        if (inputCount != tensors.Length)
        {
            throw new ArgumentException("Number of tensors does not match the number of tensor names in the einsum expression.");
        }
        foreach (var name in tensorNames)
        {
            foreach (var c in name)
            {
                symbols.Add(c);
            }
        }
        symbolTable = [.. symbols];
        dimCount = symbolTable.Length;
        symbolToIndex = symbolTable.Index().ToDictionary(
            kvp => kvp.Item, kvp => kvp.Index);
        used = new bool[inputCount + 1, dimCount];
        strides = new int[inputCount, dimCount];
        dimLengths = new int[dimCount];
        this.tensors = tensors.ToArray();
        nodes = new EinsumNode[inputCount];
        for (int i = 0; i < inputCount; i++)
        {
            var tensor = tensors[i];
            var tensorLengths = tensor.Lengths;
            var tensorStrides = tensor.Strides;
            var rank = tensor.Rank;
            var tensorName = tensorNames[i];
            if (rank != tensorName.Length)
            {
                throw new ArgumentException($"Tensor '{tensorNames[i]}' rank " +
                    $"does not match the number of symbols in the einsum expression.");
            }
            SingleIndice[] tensorIndices =
                new SingleIndice[rank];
            for (int iDim = 0; iDim < rank; iDim++)
            {
                ref var indice = ref tensorIndices[iDim];
                indice = new(tensorLengths[iDim], tensorStrides[iDim]);
                var dimChar = tensorName[iDim];
                var dimIndex = symbolToIndex[dimChar];
                if (used[i, dimIndex])
                {
                    strides[i, dimIndex] += tensorStrides[iDim];
                }
                else
                {
                    used[i, dimIndex] = true;
                    strides[i, dimIndex] = tensorStrides[iDim];
                }
                if (dimLengths[dimIndex] == 0)
                {
                    dimLengths[dimIndex] = tensorLengths[iDim];
                }
                else if (dimLengths[dimIndex] != tensorLengths[iDim])
                {
                    //TODO: For broadcast
                    throw new ArgumentException($"Tensor '{tensorNames[i]}' " +
                        $"has conflicting lengths for dimension '{dimChar}'.");
                }
            }
            nodes[i] = new InputNode(tensorName, i, tensorIndices);
        }
        tensorSize = new int[inputCount];
        for (int i = 0; i < inputCount; i++)
        {
            int size = 0;
            for (int j = 0; j < dimCount; j++)
            {
                if (used[i, j])
                    size *= dimLengths[j];
            }
            tensorSize[i] = size;
        }

        for (int iDim = 0; iDim < target.Length; iDim++)
        {
            var dimChar = target[iDim];
            if (!symbolToIndex.TryGetValue(dimChar, out int dimIndex))
            {
                throw new ArgumentException($"Target dimension '{dimChar}' " +
                    $"is not present in the input tensors.");
            }
            if (used[inputCount, dimIndex])
            {
                throw new ArgumentException($"Target dimension '{dimChar}' " +
                    $"is already used by an input tensor.");
            }
            else
            {
                used[inputCount, dimIndex] = true;
            }
        }
        size = 1;
        for (int i = 0; i < dimCount; i++)
            size *= dimLengths[i];
        this.expression = "";
    }

    private EinsumState(EinsumState parent, int left, int right, string name)
    {
        this.parent = parent;
        parent.child = this;
        child = null;
        strides = parent.strides;
        used = parent.used;
        symbolTable = parent.symbolTable;
        symbolToIndex = parent.symbolToIndex;
        dimLengths = parent.dimLengths;
        dimCount = parent.dimCount;
        inputCount = parent.inputCount - 1;
        target = parent.target;
        this.left = left;
        this.right = right;
        tensorSize = new int[inputCount];
        tensorNames = new string[inputCount];
        tensors = new MArray[inputCount];
        nodes = parent.nodes;
        int i = 0;
        for (; i < left; i++)
        {
            tensorNames[i] = parent.tensorNames[i];
            tensorSize[i] = parent.tensorSize[i];
            tensors[i] = parent.tensors[i];
            for (int j = 0; j < dimCount; j++)
            {
                used[i, j] = parent.used[i, j];
            }
        }

        tensorNames[left] = name;
        int stride = 1;
        for (int iChar = 0; iChar < dimCount; iChar++)
        {
            used[left, iChar] = false;
        }
        for (int iChar = name.Length - 1; iChar >= 0; iChar--)
        {
            var dimChar = name[iChar];
            var dimIndex = symbolToIndex[dimChar];
            used[left, dimIndex] = true;
            strides[left, dimIndex] = stride;
            stride *= dimLengths[dimIndex];
        }
        tensors[i] = null;
        tensorSize[i] = stride;
        i++;

        for (; i < right; i++)
        {
            tensorNames[i] = parent.tensorNames[i];
            tensorSize[i] = parent.tensorSize[i];
            tensors[i] = parent.tensors[i];
            for (int j = 0; j < dimCount; j++)
            {
                used[i, j] = parent.used[i, j];
                strides[i, j] = parent.strides[i, j];
            }
        }
        for (; i < inputCount; i++)
        {
            tensorNames[i] = parent.tensorNames[i + 1];
            tensorSize[i] = parent.tensorSize[i + 1];
            tensors[i] = parent.tensors[i + 1];
            nodes[i] = nodes[i + 1];
            for (int j = 0; j < dimCount; j++)
            {
                used[i, j] = parent.used[i + 1, j];
                strides[i, j] = parent.strides[i + 1, j];
            }
        }
        {
            for (int j = 0; j < dimCount; j++)
            {
                used[i, j] = parent.used[i + 1, j];
                used[i + 1, j] = false;
            }
        }


        size = 1;
        for (i = 0; i < dimCount; i++)
        {
            for (int j = 0; j < inputCount + 1; j++)
            {
                if (used[j, i])
                {
                    size *= dimLengths[i];
                    break;
                }
            }
        }
        cost = 2;
        for (i = 0; i < dimCount; i++)
        {
            if (parent.used[left, i] || parent.used[right, i])
            {
                cost *= dimLengths[i];
            }
        }
        expression =
            $"{parent.tensorNames[left]},{parent.tensorNames[right]}->{name}";
    }

    private EinsumState(EinsumState other)
    {
        parent = other.parent;
        child = other.child;
        strides = other.strides;
        used = other.used;
        symbolTable = other.symbolTable;
        symbolToIndex = other.symbolToIndex;
        dimLengths = other.dimLengths;
        dimCount = other.dimCount;
        inputCount = other.inputCount;
        target = other.target;
        left = other.left;
        right = other.right;
        tensorSize = other.tensorSize;
        tensorNames = other.tensorNames;
        tensors = other.tensors;
        size = other.size;
        cost = other.cost;
        expression = other.expression;
        nodes = other.nodes;
    }

    public static double SizeAlpha { get; set; } = 1;
    public static double FlopsAlpha { get; set; } = 0;

    public void TryContract(int left, int right,
        double sizeAlpha, double flopsAlpha,
        out double cost, out string outputName)
    {
        Span<bool> outputUsed = stackalloc bool[dimCount];
        Span<bool> thisRemoved = stackalloc bool[dimCount];
        Span<char> outputNameSpan = stackalloc char[dimCount];
        int outputUsedCount = 0;

        int leftSize = tensorSize[left];
        int rightSize = tensorSize[right];
        int resultSize = 1;
        int flops = 1;
        for (int i = 0; i < dimCount; i++)
        {
            if (!used[left, i] && !used[right, i])
                continue;
            flops *= dimLengths[i];
            bool otherUsed = false;
            for (int j = 0; j < inputCount + 1; j++)
            {
                if (j == left || j == right)
                    continue;

                if (used[j, i])
                {
                    otherUsed = true;
                    break;
                }
            }

            if (otherUsed)
            {
                outputUsed[i] = true;
                resultSize *= dimLengths[i];
                outputNameSpan[outputUsedCount++] = symbolTable[i];
            }
            else
                thisRemoved[i] = true;
        }

        outputName = outputNameSpan[..outputUsedCount].ToString();
        cost = resultSize - sizeAlpha * (leftSize + rightSize)
            + flopsAlpha * flops;
    }

    public EinsumState? Step()
    {
        if (inputCount == 1)
        {
            int index = -1;
            for (int i = 0; i < inputCount; i++)
            {
                if (nodes[i] is not null)
                {
                    if (index != -1)
                        throw new InvalidOperationException(
                            "Something error happened!");
                    index = i;
                }
            }
            if (index == -1)
                throw new InvalidOperationException(
                    "Something error happened!");
            var node = nodes[index];
            nodes[index] = null;
            var output = new OutputNode
                (target, node!);
            nodes[0] = output;
            return null;
        }

        List<Candidate> candidateSet
            = new(inputCount * (inputCount - 1) / 2);
        for (int left = 0; left < inputCount - 1; left++)
        {
            for (int right = left + 1; right < inputCount; right++)
            {
                TryContract(left, right, SizeAlpha, FlopsAlpha,
                    out var cost, out var name);
                candidateSet.Add(new Candidate
                    (left, right, cost, name));
            }
        }
        var candidate = GetCandidate(candidateSet);
        var (candiLeft, candiRight, _, candiName) = candidate;
        nodes[candiLeft] = new ContractedNode
            (candiName, nodes[candiLeft]!, nodes[candiRight]!);
        nodes[candiRight] = null;
        return new(this, candiLeft, candiRight, candiName);
    }

    public Candidate GetCandidate(List<Candidate> candidateSet)
    {
        return candidateSet
            .OrderBy(c => c.Cost)
            .ThenBy(c => c.Left + right)
            .First();
    }

    public record struct Candidate
        (int Left, int Right, double Cost, string Name)
    {
        public readonly void Deconstruct(
            out int candiLeft, out int candiRight,
            out double cost, out string candiName)
        {
            candiLeft = Left;
            candiRight = Right;
            cost = Cost;
            candiName = Name;
        }
    }

    public OutputNode Parse()
    {
        var current = this;
        while (true)
        {
            current = current.Step();
            if (current == null)
                break;
        }
        return (OutputNode)nodes[0]!;
    }

    internal EinsumState Fork()
        => new(this);
}