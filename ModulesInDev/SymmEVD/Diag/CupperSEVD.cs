using DDLA.BLAS.Managed;
using System.Diagnostics;

namespace ModulesInDev.SymmEVD.Diag;

public class CupperSEVD(VectorView d,
    VectorView e, MatrixView Q,
    double tol = 1e-16, int maxIter = 32)
{
    public double Tol { get; private set; } = tol;

    public int MaxIter { get; } = maxIter;

    public int TotalIter { get; private set; } = 0;

    public VectorView d { get; } = d;

    public VectorView e { get; } = e;

    public MatrixView Q { get; set; } = Q;

    public void Kernel()
    {
        if (d.Length == 1)
        {
            return;
        }
        else if (d.Length == 2)
        {
            var franc = new FrancisQRSEVD(d, e, Q, Tol, MaxIter);
            franc.Kernel();
        }
        else
        {
            var T = MatrixView.Diagonals(d);
            for (int i = 0; i < e.Length; i++)
            {
                T[i, i + 1] = T[i + 1, i] = e[i];
            }
            var QOrig = Q.Clone();
            Q.Clear();
            int start = 0, end;
            for (int i = 0; i < e.Length; i++)
            {
                if (Math.Abs(e[i]) <=
                    Tol * (Math.Abs(d[i]) + Math.Abs(d[i + 1])) + 1e-100)
                {
                    end = i + 1;
                    SolveSubproblem(d[start..end], e[start..i],
                        Q[start..end, start..end]);
                    start = end;
                }
            }
            SolveSubproblem(d[start..], e[start..], Q[start.., start..]);

            var rebuild = Q * MatrixView.Diagonals(d) * Q.T;
            var diff = T - rebuild;
            var norm = diff.NrmF();


            (QOrig * Q).CopyTo(Q);
        }

    }

    public void SolveSubproblem(VectorView d, VectorView e, MatrixView Q)
    {
        CupperSEVDSub cupper = new(d, e, Tol, MaxIter);
        cupper.Kernel();
        cupper.Q.CopyTo(Q);
    }
}

public partial class CupperSEVDSub(VectorView d, VectorView e,
    double tol = 1e-16, int maxIter = 32)
{
    public double Tol { get; } = tol;

    public int MaxIter { get; } = maxIter;

    public VectorView d { get; } = d;

    public VectorView e { get; } = e;

    public MatrixView Q { get; set; } = MatrixView.Eyes(d.Length, colMajor: true);

    public void Kernel()
    {
        double scale = Math.Max(d.MaxAbs(), e.MaxAbs());
        d.InvScaledBy(scale);
        e.InvScaledBy(scale);
        if (d.Length <= 24) Leaf();
        else Inner();
        d.ScaledBy(scale);
        e.ScaledBy(scale);
    }

    public void Leaf()
    {
        var fran = new FrancisQRSEVD(d, e, Q, Tol, MaxIter);
        fran.Kernel();
    }

    public void Inner()
    {
        Divide();
        Merge();
    }

    MatrixView T { get; set; }

    private int LeftSize { get; set; }

    private double Rho { get; set; }

    private VectorView z { get; set; }

    private MatrixView LeftQ { get; set; }

    private MatrixView RightQ { get; set; }

    public MatrixView QCopy { get; set; } =
        MatrixView.Create(d.Length, d.Length, colMajor: true);

    private void Divide()
    {
        T = MatrixView.Diagonals(d);
        for (int i = 0; i < e.Length; i++)
        {
            T[i, i + 1] = T[i + 1, i] = e[i];
        }
        LeftSize = d.Length / 2;

        var leftD = d[..LeftSize];
        var leftE = e[..(LeftSize - 1)];

        var rightD = d[LeftSize..];
        var rightE = e[LeftSize..];

        var bm = e[LeftSize - 1];
        Rho = Math.Abs(bm);
        leftD[^1] -= Rho;
        rightD[0] -= Rho;

        var leftSEVD = new CupperSEVDSub(leftD, leftE, Tol, MaxIter);
        var rightSEVD = new CupperSEVDSub(rightD, rightE, Tol, MaxIter);
        leftSEVD.Kernel();
        rightSEVD.Kernel();

        LeftQ = leftSEVD.Q;
        RightQ = rightSEVD.Q;
    }

    MatrixView Rank1 { get; set; }

    private void Merge()
    {
        var bm = e[LeftSize - 1];
        z = VectorView.Create(d.Length);
        LeftQ[^1, ..].CopyTo(z[..LeftSize]);
        RightQ[0, ..].CopyTo(z[LeftSize..]);
        LeftQ.CopyTo(Q[..LeftSize, ..LeftSize]);
        RightQ.CopyTo(Q[LeftSize.., LeftSize..]);
        if (bm < 0)
        {
            z[..LeftSize].ScaledBy(-1);
        }
        var normz = z.NrmF();
        Rho *= normz * normz;
        z.Normalized();

        Deflate();
        CalcEigenValuesAndVectors();
    }

    private DCResult[]? Values { get; set; }

    private int K { get; set; }

    private void Deflate()
    {
        var n = d.Length;
        var i = 0;
        Values = new DCResult[n];
        for (; i < LeftSize; i++)
        {
            Values[i] = new DCResult(this, i, ResultType.FromUpper);
        }
        for (; i < n; i++)
        {
            Values[i] = new DCResult(this, i, ResultType.FromLower);
        }

        Array.Sort(Values, (a, b) => a.d.CompareTo(b.d));

        int left = 0;
        int current = 1;
        int right = n;
        while (current < right)
        {
            ref var valueLeft = ref Values[left];
            ref var valueCurrent = ref Values[current];
            if (Rho * Math.Abs(valueCurrent.z) <= Tol)
            {
                valueCurrent.z = 0;
                valueCurrent.Type = ResultType.TooSmall;
                right--;
                var currentValue = valueCurrent;
                for (i = current; i < right; i++)
                {
                    Values[i] = Values[i + 1];
                }
                Values[right] = currentValue;
            }
            else
            {
                var a = valueLeft.z;
                var b = valueCurrent.z;
                var tau = double.Hypot(a, b);
                var c = a / tau;
                var s = -b / tau;
                var t = valueCurrent.d - valueLeft.d;
                if (Rho * Math.Abs(t * c * s) <= Tol)
                {
                    Debug.Assert(Math.Abs(c * c + s * s - 1.0) < 1e-12);
                    BlasProvider.Rot(valueLeft.OrigEigenVector,
                        valueCurrent.OrigEigenVector, (c, s));
                    valueLeft.z = tau;
                    valueCurrent.z = 0.0;
                    if (valueLeft.Type == ResultType.FromLower &&
                        valueCurrent.Type == ResultType.FromUpper)
                    {
                        valueCurrent.Type = ResultType.TooCloseAndCross;
                    }
                    right--;
                    var currentValue = valueCurrent;
                    for (i = current; i < right; i++)
                    {
                        Values[i] = Values[i + 1];
                    }
                    Values[right] = currentValue;
                }
                else
                {
                    left++;
                    current++;
                }
            }
        }
        K = right;
        for (i = 0; i < n; i++)
        {
            ref var valueCurrent = ref Values[i];
            valueCurrent.OrigEigenVector.CopyTo(QCopy.GetColumn(i));
            d[i] = valueCurrent.d;
            z[i] = valueCurrent.z;
        }
        QCopy.CopyTo(Q);
    }

    public MatrixView Deltas { get; set; }

    private void CalcEigenValuesAndVectors()
    {
        Rank1 = MatrixView.Diagonals(d);
        Rank1.Rank1(Rho, z, z);

        var rebuild = Q * Rank1 * Q.T;
        var diff = T - rebuild;
        var norm = diff.NrmF();

        Deltas = MatrixView.Create(K, K, colMajor: true);
        var invoker = new SecularEquationSolver(this);
        invoker.Invoke();
        var new_z = z[..K].Clone();
        //for (int i = 0; i < K; i++)
        //{
        //    var zi = Deltas[i, i] / Rho;
        //    var di = d[i];
        //    for (int j = 0; j < i; j++)
        //    {
        //        zi *= Deltas[j, i] / (di - d[j]);
        //    }
        //    for (int j = i + 1; j < K; j++)
        //    {
        //        zi *= Deltas[j, i] / (di - d[j]);
        //    }
        //    zi = Math.Abs(zi);
        //    var orig = new_z[i];
        //    var newz = Math.Sqrt(zi);
        //    new_z[i] = Math.CopySign(newz, orig);
        //}
        new_z.Normalized();
        QCopy.Clear();
        for (int i = 0; i < K; i++)
        {
            d[i] = Values[i].lambda;
            var col = QCopy.GetColumn(i);
            for (int j = 0; j < K; j++)
            {
                col[j] = new_z[j] / Deltas[j, i];
            }
            col.Normalized();
        }
        for (int i = K; i < d.Length; i++)
        {
            QCopy[i, i] = 1;
        }

        rebuild = QCopy * MatrixView.Diagonals(d) * QCopy.T;
        diff = Rank1 - rebuild;
        norm = diff.NrmF();
        if (norm >= 1e-10)
        {
            Console.WriteLine("Found error.");
            for (int i = 0; i < d.Length; i++)
            {
                for (int j = 0; j < d.Length; j++)
                {
                    if (Math.Abs(diff[i, j]) > 1e-10)
                    {
                        Console.WriteLine($"diff[{i}, {j}] = {diff[i, j]}");
                    }
                }
            }
        }
        (Q * QCopy).CopyTo(Q);
    }

    enum ResultType
    {
        FromUpper = 0,
        FromLower,
        TooCloseAndCross,
        TooSmall
    }

    struct DCResult
    {
        public DCResult(CupperSEVDSub orig, int origIndex, ResultType type)
        {
            OrigEigenVector = orig.Q.GetColumn(origIndex);
            d = orig.d[origIndex];
            z = orig.z[origIndex];
            lambda = d;
            OrigIndex = origIndex;
            Type = type;
        }

        public VectorView OrigEigenVector { get; set; }

        public double d { get; }

        public double z { get; set; }

        public double lambda { get; set; }

        public int OrigIndex { get; }

        public ResultType Type { get; set; }

        public override string ToString()
        {
            return $"d={d}, z={z}, lambda={lambda}";
        }
    }
}
