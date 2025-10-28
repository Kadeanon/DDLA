using System.Diagnostics.CodeAnalysis;
using System.Numerics;

namespace DDLA.UFuncs;

/// <summary>
/// Represents an operator that can be applied to the elements of a tensor.
/// </summary>
public interface IOperator
{
    /// <summary>
    /// Gets a value indicating whether the operation can be vectorized.
    /// </summary>
    static virtual bool IsVectorizable => false;

    static virtual bool ShouldManualInitialize => false;
}

/// <summary>
/// Represents an operator that performs an operation with a single parameter.
/// </summary>
/// <typeparam name="T">The type of the parameter.</typeparam>
/// <typeparam name="TResult">The type of the result.</typeparam>
public interface IUnaryOperator<T, TResult>
    : IOperator
    where T : struct
    where TResult : struct
{
    /// <summary>
    /// Performs the unary operation using the specified value.
    /// </summary>
    /// <param name="x">The value to use.</param>
    /// <returns>The result of using the operation.</returns>
    abstract TResult Invoke(T x);

    /// <summary>
    /// Performs the unary operation using the specified vector.
    /// </summary>
    /// <param name="x">The vector to use.</param>
    /// <returns>The result of using the operation.</returns>
    virtual Vector<TResult> Invoke(ref readonly Vector<T> x)
        => throw new NotSupportedException();
}

/// <summary>
/// Represents an operator that performs an operation with two parameters.
/// </summary>
/// <typeparam name="T1">The type of the first parameter.</typeparam>
/// <typeparam name="T2">The type of the second parameter.</typeparam>
/// <typeparam name="TResult">The type of the result.</typeparam>
public interface IBinaryOperator<T1, T2, TResult>
    : IOperator
    where T1 : struct
    where T2 : struct
    where TResult : struct
{
    /// <summary>
    /// Performs the binary operation using the specified values.
    /// </summary>
    /// <param name="x">The first value to use.</param>
    /// <param name="y">The second value to use.</param>
    /// <returns>The result of using the operation.</returns>
    abstract TResult Invoke(T1 x, T2 y);

    /// <summary>
    /// Performs the binary operation using the specified vectors.
    /// </summary>
    /// <param name="x">The first vector to use.</param>
    /// <param name="y">The second vector to use.</param>
    /// <returns>The result of using the operation.</returns>
    virtual Vector<TResult> Invoke(ref readonly Vector<T1> x, ref readonly Vector<T2> y)
        => throw new NotSupportedException();
}

/// <summary>
/// Represents an operator that performs an operation with three parameters.
/// </summary>
/// <typeparam name="T1">The type of the first parameter.</typeparam>
/// <typeparam name="T2">The type of the second parameter.</typeparam>
/// <typeparam name="T3">The type of the third parameter.</typeparam>
/// <typeparam name="TResult">The type of the result.</typeparam>
public interface ITernaryOperator<T1, T2, T3, TResult>
    : IOperator
    where T1 : struct
    where T2 : struct
    where T3 : struct
    where TResult : struct
{
    /// <summary>
    /// Performs the ternary operation using the specified values.
    /// </summary>
    /// <param name="x">The first value to use.</param>
    /// <param name="y">The second value to use.</param>
    /// <param name="z">The third value to use.</param>
    /// <returns>The result of using the operation.</returns>
    abstract TResult Invoke(T1 x, T2 y, T3 z);

    /// <summary>
    /// Performs the ternary operation using the specified vectors.
    /// </summary>
    /// <param name="x">The first vector to use.</param>
    /// <param name="y">The second vector to use.</param>
    /// <param name="z">The third vector to use.</param>
    /// <returns>The result of using the operation.</returns>
    virtual Vector<TResult> Invoke(ref readonly Vector<T1> x, ref readonly Vector<T2> y, ref readonly Vector<T3> z)
        => throw new NotSupportedException();
}

/// <summary>
/// Represents an aggregation operator that returns a value.
/// </summary>
/// <typeparam name="T">The type of the source elements.</typeparam>
/// <typeparam name="TResult">The type of the result value.</typeparam>
public interface IAggregationOperator<T, TResult>
    : IOperator
    where T : struct
    where TResult : struct
{
    /// <summary>
    /// Gets the seed value used to initialize the aggregation.
    /// </summary>
    /// <returns>The seed value.</returns>
    static abstract TResult Seed { get; }

    /// <summary>
    /// Gets the result of the aggregation.
    /// </summary>
    TResult Result { get; }

    /// <summary>
    /// Reduce a scalar to the inner result.
    /// </summary>
    /// <param name="val">The value to be reduce.</param>
    void Combine(T val);

    /// <summary>
    /// Reduce a vectorized sub-result to the inner result.
    /// </summary>
    /// <param name="subResult">A vector of values</param>
    void Combine(ref readonly Vector<TResult> subResult);

    /// <summary>
    /// Aggregate a vector to a existing vector result.
    /// The vector result should be combined 
    /// to the inner result by <see cref="Combine(ref readonly Vector{TResult})"/>.
    /// </summary>
    /// <param name="subResult">The first vector with preview result.</param>
    /// <param name="val">The second vector to reduced.</param>
    /// <returns>The result of using the operation.</returns>
    virtual Vector<TResult> Invoke(ref readonly Vector<TResult> subResult, ref readonly Vector<T> val)
            => throw new NotSupportedException();
}