namespace DDLA.Misc;

public readonly struct Slice
{
    readonly Index index;
    readonly Range range;
    readonly bool isIndex;

    public Slice(Index index)
    {
        this.index = index;
        range = default;
        isIndex = true;
    }

    public Slice(Range range)
    {
        index = default;
        this.range = range;
        isIndex = false;
    }

    public static implicit operator Slice(Index index)
        => new(index);

    public static implicit operator Slice(Range range)
        => new(range);

    public static implicit operator Slice(int index)
        => new(index: index);

    public readonly Index Index => index;

    public readonly Range Range => range;

    public readonly bool IsIndex => isIndex;
}
