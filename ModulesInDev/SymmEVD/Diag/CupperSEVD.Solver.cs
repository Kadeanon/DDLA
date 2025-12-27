using DDLA.Utilities;
using System.Runtime.CompilerServices;

namespace ModulesInDev.SymmEVD.Diag;

public partial class CupperSEVDSub
{
    readonly struct SecularEquationSolver(CupperSEVDSub orig) : IActionEX
    {

        private const int MaxIter = 30;
        // machine epsilon for IEEE754 double: 2^-52
        private const double MachineEps = 2.2204460492503131e-16;

        private CupperSEVDSub Orig { get; } = orig;

        private int K { get; } = orig.K;

        private VectorView d { get; } = orig.d[..orig.K];

        private VectorView z { get; } = orig.z[..orig.K];

        private MatrixView Deltas { get; } = orig.Deltas;

        private double Rho { get; } = orig.Rho;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private VectorView DeltaColumn(int index) => Deltas.GetColumn(index);

        public void Invoke(bool parallel = true)
        {
            if (parallel)
            {
                ParallelHelperEX.For(0, K, this, 16, Environment.ProcessorCount / 2);
            }
            else
            {
                for (int i = 0; i < K; i++)
                {
                    Invoke(i);
                }
            }
        }

        public void Invoke(int i)
        {
            ref var value = ref Orig.Values[i];
            value.lambda = SolveRoot(i);
        }

        private double SolveRoot(int rootIndex)
        {
            System.Diagnostics.Trace.Assert(K >= 3);
            if (Rho <= 0.0) throw new ArgumentOutOfRangeException(nameof(Rho), "This solver assumes rho > 0.");

            VectorView delta = DeltaColumn(rootIndex);

            return rootIndex == K - 1
                ? SolveLastRoot(delta)
                : SolveInteriorRoot(rootIndex, delta);
        }

        // ----------------------------
        // Helpers: sum z^2/delta and derivative sum (z/delta)^2
        // Keeps the same "erretm accumulation" style as the original.
        // ----------------------------
        private static void AccumulateForward(
            VectorView z, VectorView delta, int endExclusive,
            out double sum, out double dsum, out double errCum)
        {
            sum = 0.0;
            dsum = 0.0;
            errCum = 0.0;

            for (int j = 0; j < endExclusive; j++)
            {
                double t = z[j] / delta[j];
                sum += z[j] * t;
                dsum += t * t;
                errCum += sum;
            }

            errCum = Math.Abs(errCum);
        }

        private static void AccumulateBackward(
            VectorView z, VectorView delta, int startInclusive,
            out double sum, out double dsum, out double errCum)
        {
            sum = 0.0;
            dsum = 0.0;
            errCum = 0.0;

            for (int j = z.Length - 1; j >= startInclusive; j--)
            {
                double t = z[j] / delta[j];
                sum += z[j] * t;
                dsum += t * t;
                errCum += sum;
            }

            errCum = Math.Abs(errCum);
        }

        private static void ShiftAll(VectorView deltas, double eta)
        {
            foreach (ref var delta in deltas)
                delta -= eta;
        }

        private static void ShiftAllTo(VectorView d, double eta, double eta2, 
            VectorView deltas)
        {
            var src = d.GetEnumerator();
            foreach (ref var delta in deltas)
            {
                src.MoveNext();
                delta = src.Current - eta - eta2;
            }
        }

        // ============================================================
        // Branch A: last root (rootIndex == K-1), lambda in (d[K-1], +inf)
        // ============================================================
        private double SolveLastRoot(VectorView delta)
        {
            double rhoInv = 1.0 / Rho;

            int k = K;
            int last = k - 1;

            double midpt = Rho / 2.0;

            // base is d[last]
            double baseD = d[last];

            // delta = d - baseD - midpt
            ShiftAllTo(d, baseD, midpt, delta);

            // psi = sum_{j=0..k-3} z^2/delta
            double psi = 0.0;
            for (int j = 0; j < k - 2; j++)
                psi += z[j] * z[j] / delta[j];

            double zm2 = z[^2];
            double zm1 = z[^1];
            double dm2 = d[^2];
            double dm1 = d[^1];

            double c = rhoInv + psi;
            double w = c + zm2 * zm2 / delta[^2] + zm1 * zm1 / delta[^1];

            double tau, tauLower, tauUpper;

            if (w <= 0.0)
            {
                // choose tau in [midpt, rho]
                double temp = zm2 * zm2 / (dm1 - dm2 + Rho) + zm1 * zm1 / Rho;
                if (c <= temp)
                {
                    tau = Rho;
                }
                else
                {
                    double del = dm1 - dm2;
                    double a = -c * del + zm2 * zm2 + zm1 * zm1;
                    double b = zm1 * zm1 * del;

                    if (a < 0.0)
                        tau = b * 2.0 / (Math.Sqrt(a * a + b * 4.0 * c) - a);
                    else
                        tau = (a + Math.Sqrt(a * a + b * 4.0 * c)) / (c * 2.0);
                }

                tauLower = midpt;
                tauUpper = Rho;
            }
            else
            {
                // choose tau in [0, midpt]
                double del = dm1 - dm2;
                double a = -c * del + zm2 * zm2 + zm1 * zm1;
                double b = zm1 * zm1 * del;

                if (a < 0.0)
                    tau = b * 2.0 / (Math.Sqrt(a * a + b * 4.0 * c) - a);
                else
                    tau = (a + Math.Sqrt(a * a + b * 4.0 * c)) / (c * 2.0);

                tauLower = 0.0;
                tauUpper = midpt;
            }

            // delta = d - d[last] - tau
            ShiftAllTo(d, d[last], tau, delta);

            // Evaluate psi/dpsi and phi/dphi for last-root formula
            // Here, "psi" is over j < last; "phi" is only the last pole term (zm1)
            AccumulateForward(z, delta, last, out psi, out double dpsi, out double errPsi);

            double tLast = zm1 / delta[^1];
            double phi = zm1 * tLast;
            double dphi = tLast * tLast;

            double erretm = (-phi - psi) * 8.0 + errPsi - phi + rhoInv + Math.Abs(tau) * (dpsi + dphi);
            w = rhoInv + phi + psi;

            if (Math.Abs(w) <= MachineEps * erretm)
                return d[last] + tau;

            // tighten bracket
            if (w <= 0.0) tauLower = Math.Max(tauLower, tau);
            else tauUpper = Math.Min(tauUpper, tau);

            // one "startup" update, then iterate
            for (int iter = 2; iter <= MaxIter; iter++)
            {
                // compute eta using the same quadratic-like formula as your original
                double cStep = w - delta[^2] * dpsi - delta[^1] * dphi;
                double aStep = (delta[^2] + delta[^1]) * w - delta[^2] * delta[^1] * (dpsi + dphi);
                double bStep = delta[^2] * delta[^1] * w;

                double eta;
                if (cStep < 0.0) cStep = Math.Abs(cStep);

                if (cStep == 0.0)
                {
                    eta = tauUpper - tau;
                }
                else if (aStep >= 0.0)
                {
                    double disc = aStep * aStep - bStep * 4.0 * cStep;
                    eta = (aStep + Math.Sqrt(Math.Abs(disc))) / (cStep * 2.0);
                }
                else
                {
                    double disc = aStep * aStep - bStep * 4.0 * cStep;
                    eta = bStep * 2.0 / (aStep - Math.Sqrt(Math.Abs(disc)));
                }

                // fallback to Newton step if sign wrong
                if (w * eta > 0.0)
                    eta = -w / (dpsi + dphi);

                double tauNew = tau + eta;

                // keep inside [tauLower, tauUpper]
                if (tauNew > tauUpper || tauNew < tauLower)
                    eta = (w < 0.0) ? (tauUpper - tau) / 2.0 : (tauLower - tau) / 2.0;

                ShiftAll(delta, eta);
                tau += eta;

                // reevaluate
                AccumulateForward(z, delta, last, out psi, out dpsi, out errPsi);

                tLast = zm1 / delta[^1];
                phi = zm1 * tLast;
                dphi = tLast * tLast;

                erretm = (-phi - psi) * 8.0 + errPsi - phi + rhoInv + Math.Abs(tau) * (dpsi + dphi);
                w = rhoInv + phi + psi;

                if (Math.Abs(w) <= MachineEps * erretm)
                    return d[last] + tau;

                if (w <= 0.0) tauLower = Math.Max(tauLower, tau);
                else tauUpper = Math.Min(tauUpper, tau);
            }

            LinalgException.ThrowConvergenceFailed(nameof(SolveLastRoot));
            return d[last] + tau;
        }

        // ============================================================
        // Branch B: interior root, lambda in (d[i], d[i+1])
        // ============================================================
        private double SolveInteriorRoot(int index, VectorView delta)
        {
            double rhoInv = 1.0 / Rho;
            int k = K;

            double di = d[index];
            double del = d[index + 1] - di;
            double midpt = del / 2.0;

            // delta = d - di - midpt
            ShiftAllTo(d, di, midpt, delta);

            // psi over left part (j < index)
            double psi = 0.0;
            for (int j = 0; j < index; j++)
                psi += z[j] * z[j] / delta[j];

            // phi over right part (j > index+1)
            double phi = 0.0;
            for (int j = k - 1; j > index + 1; --j)
                phi += z[j] * z[j] / delta[j];

            double c = rhoInv + psi + phi;
            double w = c
                       + z[index] * z[index] / delta[index]
                       + z[index + 1] * z[index + 1] / delta[index + 1];

            bool originAtLeftPole;
            double tau, tauLower, tauUpper;

            if (w > 0.0)
            {
                originAtLeftPole = true;
                double a = c * del + z[index] * z[index] + z[index + 1] * z[index + 1];
                double b = z[index] * z[index] * del;

                if (a > 0.0)
                {
                    double disc = a * a - b * 4.0 * c;
                    tau = b * 2.0 / (a + Math.Sqrt(Math.Abs(disc)));
                }
                else
                {
                    double disc = a * a - b * 4.0 * c;
                    tau = (a - Math.Sqrt(Math.Abs(disc))) / (c * 2.0);
                }

                tauLower = 0.0;
                tauUpper = midpt;
            }
            else
            {
                originAtLeftPole = false;
                double a = c * del - z[index] * z[index] - z[index + 1] * z[index + 1];
                double b = z[index + 1] * z[index + 1] * del;

                if (a < 0.0)
                {
                    double disc = a * a + b * 4.0 * c;
                    tau = b * 2.0 / (a - Math.Sqrt(Math.Abs(disc)));
                }
                else
                {
                    double disc = a * a + b * 4.0 * c;
                    tau = -(a + Math.Sqrt(Math.Abs(disc))) / (c * 2.0);
                }

                tauLower = -midpt;
                tauUpper = 0.0;
            }

            // Choose base for delta and choose the "index2" split as in original.
            // Keep the same semantics to avoid breaking the LAPACK logic.
            int index2 = index; 
            if (originAtLeftPole)
            {
                ShiftAllTo(d, di, tau, delta);
            }
            else
            {
                ShiftAllTo(d, d[index + 1], tau, delta);

                index2++;
            }

            // Evaluate psi (j < ii2) and phi (j > ii2)
            AccumulateForward(z, delta, index2, out psi, out double dpsi, out double errPsi);
            AccumulateBackward(z, delta, index2 + 1, out phi, out double dphi, out double errPhi);

            w = rhoInv + phi + psi;

            // tripolar mode enable
            bool swtch3 = false;
            if (originAtLeftPole)
            {
                if (w < 0.0) swtch3 = true;
            }
            else
            {
                if (w > 0.0) swtch3 = true;
            }

            if (index2 == 0 || index2 == k - 1) swtch3 = false;

            // Add the central term (index2) to w and dw
            double tCenter = z[index2] / delta[index2];
            double dw = dpsi + dphi + tCenter * tCenter;
            double zOver = z[index2] * tCenter;
            w += zOver;

            double erretm = (phi - psi) * 8.0 + errPsi + errPhi
                            + rhoInv * 2.0 + Math.Abs(zOver) * 3.0
                            + Math.Abs(tau) * dw;

            if (Math.Abs(w) <= MachineEps * erretm)
                return (originAtLeftPole ? di : d[index + 1]) + tau;

            if (w <= 0.0) tauLower = Math.Max(tauLower, tau);
            else tauUpper = Math.Min(tauUpper, tau);

            double prew = w;
            bool swtch = false;

            Span<double> zz = stackalloc double[3];

            for (int iter = 2; iter <= MaxIter; iter++)
            {
                double eta;

                if (!swtch3)
                {
                    // same quadratic-like update as original (two-pole model)
                    double aStep, bStep, cStep;
                    if (!swtch)
                    {
                        if (originAtLeftPole)
                        {
                            double d2 = z[index] / delta[index];
                            cStep = w - delta[index + 1] * dw - (di - d[index + 1]) * (d2 * d2);
                        }
                        else
                        {
                            double d2 = z[index + 1] / delta[index + 1];
                            cStep = w - delta[index] * dw - (d[index + 1] - di) * (d2 * d2);
                        }
                    }
                    else
                    {
                        double t = z[index2] / delta[index2];
                        if (originAtLeftPole) dpsi += t * t;
                        else dphi += t * t;

                        cStep = w - delta[index] * dpsi - delta[index + 1] * dphi;
                    }

                    aStep = (delta[index] + delta[index + 1]) * w - delta[index] * delta[index + 1] * dw;
                    bStep = delta[index] * delta[index + 1] * w;

                    if (cStep == 0.0)
                    {
                        double aAlt = aStep;
                        if (aAlt == 0.0)
                        {
                            if (!swtch)
                            {
                                if (originAtLeftPole)
                                    aAlt = z[index] * z[index] + delta[index + 1] * delta[index + 1] * (dpsi + dphi);
                                else
                                    aAlt = z[index + 1] * z[index + 1] + delta[index] * delta[index] * (dpsi + dphi);
                            }
                            else
                            {
                                aAlt = delta[index] * delta[index] * dpsi + delta[index + 1] * delta[index + 1] * dphi;
                            }
                        }

                        eta = bStep / aAlt;
                    }
                    else if (aStep <= 0.0)
                    {
                        double disc = aStep * aStep - bStep * 4.0 * cStep;
                        eta = (aStep - Math.Sqrt(Math.Abs(disc))) / (cStep * 2.0);
                    }
                    else
                    {
                        double disc = aStep * aStep - bStep * 4.0 * cStep;
                        eta = bStep * 2.0 / (aStep + Math.Sqrt(Math.Abs(disc)));
                    }
                }
                else
                {
                    // Tripolar interpolation (three most relevant poles)
                    double temp = rhoInv + psi + phi;

                    double c0;
                    if (swtch)
                    {
                        c0 = temp - delta[index2 + 1 - 2] * dpsi - delta[index2 + 1] * dphi;
                        zz[0] = delta[index2 + 1 - 2] * delta[index2 + 1 - 2] * dpsi;
                        zz[2] = delta[index2 + 1] * delta[index2 + 1] * dphi;
                    }
                    else
                    {
                        if (originAtLeftPole)
                        {
                            double zm1 = z[index2 - 1];
                            double deltap1 = delta[index2 + 1];
                            double t = zm1 / delta[index2 - 1];
                            double t2 = t * t;
                            c0 = temp - deltap1 * (dpsi + dphi) - (d[index2 - 1] - d[index2 + 1]) * t2;
                            zz[0] = zm1 * zm1;
                            zz[2] = deltap1 * deltap1 * (dpsi - t2 + dphi);
                        }
                        else
                        {
                            double zp1 = z[index2 + 1];
                            double deltam1 = delta[index2 - 1];
                            double t = z[index2 + 1] / delta[index2 + 1];
                            double t2 = t * t;
                            c0 = temp - delta[index2 - 1] * (dpsi + dphi) - (d[index2 + 1] - d[index2 - 1]) * t2;
                            zz[0] = deltam1 * deltam1 * (dpsi + (dphi - t2));
                            zz[2] = zp1 * zp1;
                        }
                    }

                    zz[1] = z[index2] * z[index2];

                    eta = TripolarInterpolationStep(iter, originAtLeftPole, c0, 
                        delta.Slice(index2 - 1, 3), zz, w);
                }

                // if sign wrong => Newton step
                if (w * eta >= 0.0)
                    eta = -w / dw;

                double tauNew = tau + eta;
                if (tauNew > tauUpper || tauNew < tauLower)
                    eta = (w < 0.0) ? (tauUpper - tau) / 2.0 : (tauLower - tau) / 2.0;

                prew = w;

                ShiftAll(delta, eta);
                tau += eta;

                // Recompute psi/phi
                AccumulateForward(z, delta, index2, out psi, out dpsi, out errPsi);
                AccumulateBackward(z, delta, index2 + 1, out phi, out dphi, out errPhi);

                tCenter = z[index2] / delta[index2];
                dw = dpsi + dphi + tCenter * tCenter;
                zOver = z[index2] * tCenter;

                w = rhoInv + phi + psi + zOver;

                erretm = (phi - psi) * 8.0 + errPsi + errPhi + rhoInv * 2.0
                         + Math.Abs(zOver) * 3.0 + Math.Abs(tau) * dw;

                // update swtch heuristic
                if (originAtLeftPole)
                {
                    if (-w > Math.Abs(prew) / 10.0) swtch = true;
                }
                else
                {
                    if (w > Math.Abs(prew) / 10.0) swtch = true;
                }

                if (Math.Abs(w) <= MachineEps * erretm)
                    return (originAtLeftPole ? di : d[index + 1]) + tau;

                if (w <= 0.0) tauLower = Math.Max(tauLower, tau);
                else tauUpper = Math.Min(tauUpper, tau);

                if (w * prew > 0.0 && Math.Abs(w) > Math.Abs(prew) / 10.0)
                    swtch = !swtch;
            }

            LinalgException.ThrowConvergenceFailed(nameof(SolveInteriorRoot));
            return di + tau;
        }

        private static double TripolarInterpolationStep(
            int iter, bool originAtLeftPole, double c0,
            VectorView polesDeltas, Span<double> z3, double fAtTau0)
        {
            const int MAXIT = 40;

            double d0 = polesDeltas[0];
            double d1 = polesDeltas[1];
            double d2 = polesDeltas[2];

            double lbd = originAtLeftPole ? d1 : d0;
            double ubd = originAtLeftPole ? d2 : d1;

            if (fAtTau0 < 0) lbd = 0.0;
            else ubd = 0.0;

            double tau = 0.0;

            if (iter == 2)
            {
                double temp, a, b, c;

                if (originAtLeftPole)
                {
                    temp = (d2 - d1) / 2.0;
                    c = c0 + z3[0] / (d0 - d1 - temp);
                    a = c * (d1 + d2) + z3[1] + z3[2];
                    b = c * d1 * d2 + z3[1] * d2 + z3[2] * d1;
                }
                else
                {
                    temp = (d0 - d1) / 2.0;
                    c = c0 + z3[2] / (d2 - d1 - temp);
                    a = c * (d0 + d1) + z3[0] + z3[1];
                    b = c * d0 * d1 + z3[0] * d1 + z3[1] * d0;
                }

                temp = Math.Max(Math.Max(Math.Abs(a), Math.Abs(b)), Math.Abs(c));
                a /= temp; b /= temp; c /= temp;

                if (c == 0) tau = b / a;
                else if (a <= 0) tau = (a - Math.Sqrt(Math.Abs(a * a - 4 * b * c))) / (2 * c);
                else tau = 2 * b / (a + Math.Sqrt(Math.Abs(a * a - 4 * b * c)));

                if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2.0;

                // if tau hits a pole, reset to 0
                if (d0 == tau || d1 == tau || d2 == tau) tau = 0.0;
                else
                {
                    double tempEval =
                        fAtTau0
                        + tau * z3[0] / (d0 * (d0 - tau))
                        + tau * z3[1] / (d1 * (d1 - tau))
                        + tau * z3[2] / (d2 * (d2 - tau));

                    if (tempEval <= 0) lbd = tau;
                    else ubd = tau;

                    if (Math.Abs(fAtTau0) <= Math.Abs(tempEval))
                        tau = 0.0;
                }
            }

            // scaling logic and main loop preserved
            double eps = MachineEps;
            double baseVal = 2.0;
            double small1 = Math.Pow(baseVal, Math.Floor(Math.Log(eps) / Math.Log(baseVal) / 3.0));
            double sminv1 = 1.0 / small1;
            double small2 = small1 * small1;
            double sminv2 = sminv1 * sminv1;

            double tempMin = originAtLeftPole
                ? Math.Min(Math.Abs(d1 - tau), Math.Abs(d2 - tau))
                : Math.Min(Math.Abs(d0 - tau), Math.Abs(d1 - tau));

            bool scale = false;
            double sclfac = 1.0, sclinv = 1.0;

            double sd0 = d0, sd1 = d1, sd2 = d2;
            double sz0 = z3[0], sz1 = z3[1], sz2 = z3[2];

            if (tempMin <= small1)
            {
                scale = true;
                if (tempMin <= small2) { sclfac = sminv2; sclinv = small2; }
                else { sclfac = sminv1; sclinv = small1; }

                sd0 *= sclfac; sd1 *= sclfac; sd2 *= sclfac;
                sz0 *= sclfac; sz1 *= sclfac; sz2 *= sclfac;
                tau *= sclfac; lbd *= sclfac; ubd *= sclfac;
            }

            double fc = 0.0, df = 0.0, ddf = 0.0;
            {
                double t, t1, t2, t3;
                t = 1.0 / (sd0 - tau); t1 = sz0 * t; t2 = t1 * t; t3 = t2 * t; fc += t1 / sd0; df += t2; ddf += t3;
                t = 1.0 / (sd1 - tau); t1 = sz1 * t; t2 = t1 * t; t3 = t2 * t; fc += t1 / sd1; df += t2; ddf += t3;
                t = 1.0 / (sd2 - tau); t1 = sz2 * t; t2 = t1 * t; t3 = t2 * t; fc += t1 / sd2; df += t2; ddf += t3;
            }

            double f = fAtTau0 + tau * fc;
            if (f <= 0) lbd = tau; else ubd = tau;

            for (int niter = 1; niter <= MAXIT; niter++)
            {
                double t1 = originAtLeftPole ? (sd1 - tau) : (sd0 - tau);
                double t2 = originAtLeftPole ? (sd2 - tau) : (sd1 - tau);

                double a = (t1 + t2) * f - t1 * t2 * df;
                double b = t1 * t2 * f;
                double c = f - (t1 + t2) * df + t1 * t2 * ddf;

                double s = Math.Max(Math.Max(Math.Abs(a), Math.Abs(b)), Math.Abs(c));
                a /= s; b /= s; c /= s;

                double eta;
                if (c == 0) eta = b / a;
                else if (a <= 0) eta = (a - Math.Sqrt(Math.Abs(a * a - 4 * b * c))) / (2 * c);
                else eta = 2 * b / (a + Math.Sqrt(Math.Abs(a * a - 4 * b * c)));

                if (f * eta >= 0) eta = -f / df;

                tau += eta;
                if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2.0;

                // recompute
                fc = 0.0; df = 0.0; ddf = 0.0;
                double erretrm = 0.0;

                // pole hit check
                if (sd0 == tau || sd1 == tau || sd2 == tau)
                    LinalgException.ThrowConvergenceFailed(nameof(TripolarInterpolationStep));

                for (int i = 0; i < 3; i++)
                {
                    double di = (i == 0) ? sd0 : (i == 1 ? sd1 : sd2);
                    double zi = (i == 0) ? sz0 : (i == 1 ? sz1 : sz2);

                    double t = 1.0 / (di - tau);
                    double t1z = zi * t;
                    double t2z = t1z * t;
                    double t3z = t2z * t;
                    double t4 = t1z / di;

                    erretrm += Math.Abs(t4);
                    df += t2z;
                    ddf += t3z;
                    fc += t4;
                }

                f = fAtTau0 + tau * fc;
                erretrm = 8.0 * (Math.Abs(fAtTau0) + Math.Abs(tau) * erretrm) + Math.Abs(tau) * df;

                if (Math.Abs(f) <= 4 * eps * erretrm || (ubd - lbd) <= 4 * eps * Math.Abs(tau))
                    break;

                if (f <= 0) lbd = tau; else ubd = tau;
            }

            if (scale) tau *= sclinv;
            return tau;
        }
    }
}
