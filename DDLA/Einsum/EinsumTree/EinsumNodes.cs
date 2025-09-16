using DDLA.Core;
using DDLA.Misc;
using System.Diagnostics;
using System.Text;

namespace DDLA.Einsum.EinsumTree;

public abstract class EinsumNode(string name)
{
    public string Name { get; set; } = name;

    protected internal abstract MArray Kernel(ReadOnlySpan<MArray> inputs);

    protected internal abstract void Check(ReadOnlySpan<MArray> inputs);
}

public class InputNode(string symbol, int index,
    SingleIndice[] indices) : EinsumNode(symbol)
{
    public SingleIndice[] Indices { get; set; } = indices;

    public int Index { get; set; } = index;

    public override string ToString()
    {
        StringBuilder sb = new();
        var rank = Name.Length;
        sb.Append(Name);
        sb.Append(' ');
        for (int i = 0; i < rank; i++)
        {
            var c = Name[i];
            var ind = Indices[i];
            sb.Append($"{c}[{ind.Length}:{ind.Stride}]");
        }
        return sb.ToString();
    }

    protected internal override MArray Kernel(ReadOnlySpan<MArray> inputs)
    {
        return inputs[Index];
    }

    protected internal override void Check(ReadOnlySpan<MArray> inputs)
    {
        if (inputs.Length <= Index)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs),
                $"Input index {Index} is out of range for inputs of length {inputs.Length}.");
        }
    }
}

[DebuggerDisplay($"{{{nameof(Expression)}}}")]
public class ContractedNode(string symbol,
    EinsumNode left, EinsumNode right) : EinsumNode(symbol)
{
    public EinsumNode NodeLeft { get; set; } = left;

    public EinsumNode NodeRight { get; set; } = right;

    public string Expression
        => $"{NodeLeft.Name},{NodeRight.Name}->{Name}";

    protected internal override MArray Kernel(ReadOnlySpan<MArray> inputs)
    {
        var left = NodeLeft.Kernel(inputs);
        var right = NodeRight.Kernel(inputs);
        return MArray.Contract(Expression,
            left, right);
    }

    protected internal override void Check(ReadOnlySpan<MArray> inputs)
    {
        NodeLeft.Check(inputs);
        NodeRight.Check(inputs);
    }
}

public class OutputNode : EinsumNode
{
    public EinsumNode Child { get; set; }

    public OutputNode(string symbol, EinsumNode child) : base(symbol)
    {
        if (child.Name != symbol)
        {
            if (child is ContractedNode)
            {
                child.Name = symbol;
            }
        }
        Child = child;
    }

    public override string ToString()
    {
        return $"{Child.Name}->{Name}";
    }

    protected internal override MArray Kernel(ReadOnlySpan<MArray> inputs)
    {
        return Child.Kernel(inputs);
    }

    public MArray Invoke(params ReadOnlySpan<MArray> inputs)
    {
        if (inputs.Length == 0)
        {
            throw new ArgumentException("No inputs provided for the output node.");
        }
        Check(inputs);
        var result = Kernel(inputs);
        return result;
    }

    protected internal override void Check(ReadOnlySpan<MArray> inputs)
    {
        Child.Check(inputs);
    }
}