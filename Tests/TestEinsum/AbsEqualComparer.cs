using System.Diagnostics.CodeAnalysis;
using System.Numerics;

namespace Tests.TestEinsum;

internal class AbsEqualComparer<T>(T eps) : IEqualityComparer<T>
    where T : unmanaged, INumberBase<T>, IComparisonOperators<T, T, bool>
{
    public bool Equals(T x, T y)
        => T.Abs(x - y) <= eps;

    public int GetHashCode([DisallowNull] T obj)
        => obj.GetHashCode();
}
