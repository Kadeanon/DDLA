namespace DDLA.Misc;

public struct ContinuousLayer
    (SingleIndice dim, int index, bool isHead = false)
{
    public bool IsHead { get; internal set; } = isHead;
    public int Index { get; internal set; } = index;
    public int Length { get; internal set; } = dim.Length;
    public int BlockSize { get; internal set; } = dim.Length * dim.Stride;
    public int Stride { get; internal set; } = dim.Stride;

    public override readonly string ToString()
    {
        if (IsHead)
        {
            return $"Head Layer {Index}: Length={Length}, Size={BlockSize}, Stride={Stride}";
        }
        else
        {
            return $"Layer {Index}: Length={Length}, Size={BlockSize}, Stride={Stride}";
        }
    }
}
