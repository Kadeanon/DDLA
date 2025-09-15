using DDLA.Misc.Pools;
using System.Text;

namespace DDLA.Misc;

public class ContinuousInfo
{
    public int NumLayer { get; }

    public ContinuousLayer[] Layers { get; }

    public ContinuousInfo(ReadOnlySpan<int> head, bool shouldSort = true)
    {
        if (head.Length == 0)
        {
            NumLayer = 0;
            Layers = [];
            return;
        }

        int rank = head.Length / 2;
        var layers = new ContinuousLayer[rank * 2];

        Span<SingleIndice> dims = stackalloc SingleIndice[rank];
        for (int i = 0; i < rank; i++)
            dims[i] = new SingleIndice(head[i], head[rank + i]);

        if (shouldSort)
        {
            // sort by stride descending
            dims.Sort((a, b) => b.Stride.CompareTo(a.Stride));
        }

        int layerIndex = 0;
        var layersArrayIndex = 0;
        int layersHeadIndex = 0;

        var dim = dims[layersArrayIndex];
        // 初始化第一个 head layer
        ref var headLayer = ref layers[layersArrayIndex++];
        headLayer = new(dim, 0, isHead: true);
        layers[layersArrayIndex++] = new(dim, 0);


        for (int dimIndex = 1; dimIndex < rank; dimIndex++)
        {
            dim = dims[dimIndex];

            if (headLayer.Stride == dim.Length * dim.Stride)
            {
                headLayer.Length *= dim.Length;
                headLayer.Stride = dim.Stride;
            }
            else // should create new layer
            {
                layersHeadIndex = layersArrayIndex++;
                headLayer = ref layers[layersHeadIndex];
                layerIndex++;
            }
            layers[layersArrayIndex++] = new(dim, dimIndex);
        }
        Layers = layers.AsSpan(0, layersArrayIndex).ToArray();
        NumLayer = layerIndex + 1;
    }

    public void ToString(StringBuilder sb)
    {
        var head = default(ContinuousLayer);
        bool first = true;
        foreach (var layer in Layers)
        {
            if (layer.IsHead)
            {
                if (!first)
                {
                    sb.Append($"totalStep={head.Stride}  ")
                    .Append($"totalLength={head.Length}  ")
                    .Append($"totalBlock={head.BlockSize}")
                    .AppendLine();
                }
                head = layer;
                sb.AppendLine($"Layer {layer.Index}: ");
            }
            else
            {
                sb.Append($">>DimIndex: {layer.Index} -> ")
                    .Append($"step={layer.Stride}  ")
                    .Append($"length={layer.Length}  ")
                    .Append($"block={layer.BlockSize}")
                    .AppendLine();
            }
        }
        sb.Append($"totalStep={head.Stride}  ")
        .Append($"totalLength={head.Length}  ")
        .Append($"totalSize={head.BlockSize}");
    }

    public override string ToString()
    {
        using var _ = StringBuilderPool.Borrow(out var sb);
        ToString(sb);
        return sb.ToString();
    }
}
