namespace DDLA.Misc;

public readonly record struct SingleIndice(int Length, int Stride)
{ }

public readonly record struct DoubleIndice(int Length, int AStride,
    int BStride)
{
    public SingleIndice A => new(Length, AStride);

    public SingleIndice B => new(Length, BStride);

    public DoubleIndice Swap() => new(Length, BStride, AStride);
}

public readonly record struct TripleIndice(int Length, int AStride,
    int BStride, int CStride)
{
    public SingleIndice A => new(Length, AStride);
    public SingleIndice B => new(Length, BStride);
    public SingleIndice C => new(Length, CStride);

    public DoubleIndice AB => new(Length, AStride, BStride);
    public DoubleIndice AC => new(Length, AStride, CStride);
    public DoubleIndice BC => new(Length, BStride, CStride);
}
