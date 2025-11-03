using DDLA.Core;
using System.Buffers;

using ArrayHandle = DDLA.Misc.Pools.ArrayHandle
    <double, System.Buffers.ArrayPool<double>>;

namespace DDLA.Misc.Pools
{
    public class InternelPool
    {
        public static InternelPool Instance { get; }
            = new InternelPool();

        public ArrayPool<double> Pool { get; }

        private InternelPool()
        {
            Pool = ArrayPool<double>.Create();
        }

        public Vector GetVector(int length, bool init = true)
        {
            var data = Pool.Rent(length);
            var vec = new Vector(data, 0, length);
            if (init) vec.Clear();
            return vec;
        }

        public Vector GetVector(int length, double fill)
        {
            var data = Pool.Rent(length);
            var vec = new Vector(data, 0, length);
            vec.Fill(fill);
            return vec;
        }

        public Matrix GetMatrix(int rows, int cols,
            bool rowMajor = true, bool init = true)
        {
            var data = Pool.Rent(rows * cols);
            Matrix mat;
            if (rowMajor)
                mat = new(data, 0, rows, cols, cols, 1);
            else
                mat = new(data, 0, rows, cols, 1, rows);
            if (init) mat.Clear();
            return mat;
        }

        public Matrix GetMatrix(int rows, int cols,
            double fill, bool rowMajor = true)
        {
            var data = Pool.Rent(rows * cols);
            Matrix mat;
            if (rowMajor)
                mat = new(data, 0, rows, cols, cols, 1);
            else
                mat = new(data, 0, rows, cols, 1, rows);
            mat.Fill(fill);
            return mat;
        }

        public void Return(VectorView vec, bool clear = false)
        {
            Pool.Return(vec.Data, clear);
        }

        public void Return(MatrixView mat, bool clear = false)
        {
            Pool.Return(mat.Data, clear);
        }

        public static VectorHandle TakeVector(int length,
            out Vector vec, bool init = true)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, length, out var data);
            vec = new(data, 0, length);
            if (init) vec.Clear();
            return new(Instance, handle);
        }

        public static VectorHandle TakeVector(int length,
            out Vector vec, double fill)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, length, out var data);
            vec = new(data);
            vec.Fill(fill);
            return new(Instance, handle);
        }

        public static MatrixHandle TakeMatrix(int rows, int cols,
            out Matrix mat, bool rowMajor = true, bool init = true)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, rows * cols, out var data);
            if (rowMajor)
                mat = new(data, 0, rows, cols, cols, 1);
            else
                mat = new(data, 0, rows, cols, 1, rows);
            if (init) mat.Clear();
            return new(Instance, handle);
        }

        public static MatrixHandle TakeMatrix(int rows, int cols,
            out Matrix mat, double fill, bool rowMajor = true)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, rows * cols, out var data);
            if (rowMajor)
                mat = new(data, 0, rows, cols, cols, 1);
            else
                mat = new(data, 0, rows, cols, 1, rows);
            mat.Fill(fill);
            return new(Instance, handle);
        }

        public static MatrixHandle TakeMatrixView(int rows, int cols,
            out MatrixView mat, bool rowMajor = true, bool init = true)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, rows * cols, out var data);
            if (rowMajor)
                mat = new(data, 0, rows, cols, cols, 1);
            else
                mat = new(data, 0, rows, cols, 1, rows);
            if (init) mat.Clear();
            return new(Instance, handle);
        }

        public static MatrixHandle TakeMatrixView(int rows, int cols,
            out MatrixView mat, double fill, bool rowMajor = true)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, rows * cols, out var data);
            if (rowMajor)
                mat = new(data, 0, rows, cols, cols, 1);
            else
                mat = new(data, 0, rows, cols, 1, rows);
            mat.Fill(fill);
            return new(Instance, handle);
        }

        public static ArrayHandle TakeArraySegement(int length,
            out ArraySegment<double> seg, bool init = true)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, length, out var data);
            seg = new ArraySegment<double>(data, 0, length);
            if (init) seg.AsSpan().Clear();
            return handle;
        }

        public static ArrayHandle TakeArraySegement(int length,
            out ArraySegment<double> seg, double fill)
        {
            var handle = new ArrayHandle<double, ArrayPool<double>>
                (Instance.Pool, length, out var data);
            seg = new ArraySegment<double>(data, 0, length);
            seg.AsSpan().Fill(fill);
            return handle;
        }
    }

    public readonly struct VectorHandle(InternelPool pool,
         ArrayHandle handle) : IPooledHandle<Vector>
    {
        public ArrayHandle InternelHandle { get;} = handle;

        public InternelPool Pool { get; } = pool;

        public bool ReturnedToPool => InternelHandle.ReturnedToPool;

        public void Return() => InternelHandle.Return();
    }

    public readonly struct MatrixHandle(InternelPool pool,
         ArrayHandle handle) : IPooledHandle<Matrix>
    {
        public ArrayHandle InternelHandle { get; } = handle;

        public InternelPool Pool { get; } = pool;

        public bool ReturnedToPool => InternelHandle.ReturnedToPool;

        public void Return() => InternelHandle.Return();
    }

}
