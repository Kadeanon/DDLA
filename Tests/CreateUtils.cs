

namespace Tests;

internal static class CreateUtils
{
    internal static Vector CreateVector(int size)
    {
        var vector = new Vector(new double[size]);
        return vector;
    }

    internal static Vector CopyVector(Vector src)
    {
        var dst = new Vector(new double[src.Length]);
        src.CopyTo(dst);
        return dst;
    }

    internal static Vector CreateVectorRandom(int size, Random? random = null)
    {
        var vector = new Vector(new double[size]);
        random ??= Random.Shared;
        for (int i = 0; i < size; i++)
        {
            vector[i] = random.NextDouble();
        }
        return vector;
    }

    internal static Vector CreateVector(int size, double val)
    {
        var arr = new double[size];
        arr.AsSpan().Fill(val);
        var vector = new Vector(arr);
        return vector;
    }

    internal static Vector CreateVectorStride(int size, int step)
    {
        var arr = new double[size * step];
        var vector = new Vector(arr, 0, size, step: step);
        return vector;
    }

    internal static Vector CreateVectorStride(int size, int step, double val)
    {
        var arr = new double[size * step];
        arr.AsSpan().Fill(val);
        var vector = new Vector(arr, 0, size, step: step);
        return vector;
    }

    internal static Vector CreateVectorStrideRandom(int size, int step, Random? random = null)
    {
        var arr = new double[size * step];
        random ??= Random.Shared;
        var vector = new Vector(arr, 0, size, step: step);
        for (int i = 0; i < size; i++)
        {
            vector[i] = random.NextDouble();
        }
        return vector;
    }

    internal static Matrix CreateMatrix(int rows, int cols)
    {
        var arr = new double[rows * cols];
        var matrix = new Matrix(arr, 0, rows, cols);
        return matrix;
    }

    internal static Matrix CreateMatrixTrans(int rows, int cols)
    {
        var arr = new double[rows * cols];
        var matrix = new Matrix(arr, 0, rows, cols, 1, rows);
        return matrix;
    }

    internal static Matrix CopyMatrix(MatrixView src)
    {
        var dst = Matrix.Create(src.Rows, src.Cols);
        src.CopyTo(dst);
        return dst;
    }

    internal static Matrix CreateMatrixRandom(int rows, int cols, Random? random = null)
    {
        var arr = new double[rows * cols];
        random ??= Random.Shared;
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = random.NextDouble();
        }
        var matrix = new Matrix(arr, 0, rows, cols);
        return matrix;
    }

    internal static Matrix CreateMatrixTransRandom(int rows, int cols, Random? random = null)
    {
        var mat = CreateMatrixRandom(cols, rows, random);
        return mat.Transpose();
    }

    internal static Matrix CreateMatrixStrideRandom(int rows, int cols, int colStride, Random? random = null)
    {
        var arr = new double[rows * cols * colStride];
        random ??= Random.Shared;
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = random.NextDouble();
        }
        var matrix = new Matrix(arr, 0, rows, cols, cols * colStride, colStride);
        return matrix;
    }

    internal static Matrix CreateMatrix(int rows, int cols, double val, bool colMajor = false)
    {
        var arr = new double[rows * cols];
        arr.AsSpan().Fill(val);
        var matrix = colMajor ? new Matrix(arr, 0, rows, cols, 1, rows)
            : new Matrix(arr, 0, rows, cols, cols, 1);
        return matrix;
    }

    internal static Vector Range(double start, double stride, int length)
    {
        var vector = new Vector(new double[length]);
        for (int i = 0; i < length; i++)
        {
            vector[i] = start + i * stride;
        }
        return vector;
    }

    internal static Vector Range(double start, int length)
    {
        var vector = new Vector(new double[length]);
        for (int i = 0; i < length; i++)
        {
            vector[i] = start + i;
        }
        return vector;
    }

    internal static Vector Range(int length)
        => Range(1, length);
}
