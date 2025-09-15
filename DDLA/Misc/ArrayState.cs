namespace DDLA.Misc;

/// <summary>
/// An enum that represents how an array held its data.
/// </summary>
[Flags]
public enum ArrayState
{
    Default = 0,
    /// <summary>
    /// Indicate the array has an at least segment view of the data. It means the data it holds is not completely continuously distributed in memory.
    /// </summary>
    Segmented = 1,
    /// <summary>
    /// Indicate the array has a broken view of the data, which means all the strides are great than 1.
    /// </summary>
    Broken = 2,
    /// <summary>
    /// Indicate the array has a C-style view of the data. It means the stride of the last dimension is equal to 1.
    /// </summary>
    CStyle = 4,
    /// <summary>
    /// Indicate the array has a Fortran-style view of the data. It means the stride of the first dimension is equal to 1.
    /// </summary>
    FortranStyle = 8
}
