using DDLA.Misc;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.UFuncs.Operators;
using DDLA.UFuncs;
using DDLA.Misc.Flags;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
	public static void Axpy2(scalar alphax, scalar alphay, 
		in vector x, in vector y, in vector z)
	{
		TripleIndice indice = CheckIndice(x, y, z);
		Details.Axpy2V_Impl(ref x[0], alphax, alphay, ref y[0], ref z[0], indice);
	}

	public static partial class Details
	{
		public static void Axpy2V_Impl(ref scalar xHead, scalar alphax, scalar alphay,
			ref scalar yHead, ref scalar zHead, TripleIndice indice)
		{
			if (indice.Length == 0)
				return;
			
			MultiplyAddOperator<scalar> action = default;
            if (alphax == 0.0)
				UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>
					(ref xHead, alphay, ref zHead, indice.BC, action);
			else if (alphay == 0.0)
				UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>
					(ref xHead, alphax, ref zHead, indice.AC, action);
			else if ((indice.AStride == 1) && (indice.BStride == 1) && (indice.CStride == 1))
				Axpy2V_Kernel_Vector256(indice.Length, alphax, alphay, ref xHead,
					ref yHead, ref zHead);
			else
				Axpy2V_Kernel(ref xHead, alphax, alphay, ref yHead, ref zHead, indice);
		}

		public static void Axpy2V_Kernel_Vector256(int length, scalar alphax, scalar alphay, ref scalar xHead, ref scalar yHead, ref scalar zHead)
		{
			int iterStride = 4;
			int iterSize = iterStride * Vector256<scalar>.Count;
			int i = 0;
			if (length >= iterSize)
			{
				ref var xHead1 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count);
				ref var yHead1 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count);
				ref var zHead1 = ref Unsafe.Add(ref zHead, Vector256<scalar>.Count);
				ref var xHead2 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 2);
				ref var yHead2 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 2);
				ref var zHead2 = ref Unsafe.Add(ref zHead, Vector256<scalar>.Count * 2);
				ref var xHead3 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 3);
				ref var yHead3 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 3);
				ref var zHead3 = ref Unsafe.Add(ref zHead, Vector256<scalar>.Count * 3);
				Vector256<scalar> alphaVecx = Vector256.Create(alphax);
				Vector256<scalar> alphaVecy = Vector256.Create(alphay);
				for (; i <= length - iterSize; i += iterSize)
				{
					Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xHead);
					Vector256<scalar> yVec0 = Vector256.LoadUnsafe(ref yHead);
					Vector256<scalar> zVec0 = Vector256.LoadUnsafe(ref zHead);
					Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xHead1);
					Vector256<scalar> yVec1 = Vector256.LoadUnsafe(ref yHead1);
					Vector256<scalar> zVec1 = Vector256.LoadUnsafe(ref zHead1);
					Vector256<scalar> xVec2 = Vector256.LoadUnsafe(ref xHead2);
					Vector256<scalar> yVec2 = Vector256.LoadUnsafe(ref yHead2);
					Vector256<scalar> zVec2 = Vector256.LoadUnsafe(ref zHead2);
					Vector256<scalar> xVec3 = Vector256.LoadUnsafe(ref xHead3);
					Vector256<scalar> yVec3 = Vector256.LoadUnsafe(ref yHead3);
					Vector256<scalar> zVec3 = Vector256.LoadUnsafe(ref zHead3);
					if (Fma.IsSupported)
					{
						zVec0 = Fma.MultiplyAdd(xVec0, alphaVecx, zVec0);
						zVec1 = Fma.MultiplyAdd(xVec1, alphaVecx, zVec1);
						zVec2 = Fma.MultiplyAdd(xVec2, alphaVecx, zVec2);
						zVec3 = Fma.MultiplyAdd(xVec3, alphaVecx, zVec3);
						zVec0 = Fma.MultiplyAdd(yVec0, alphaVecy, zVec0);
						zVec1 = Fma.MultiplyAdd(yVec1, alphaVecy, zVec1);
						zVec2 = Fma.MultiplyAdd(yVec2, alphaVecy, zVec2);
						zVec3 = Fma.MultiplyAdd(yVec3, alphaVecy, zVec3);
					}
					else
					{
						zVec0 += xVec0 * alphaVecx + yVec0 * alphaVecy;
						zVec1 += xVec1 * alphaVecx + yVec1 * alphaVecy;
						zVec2 += xVec2 * alphaVecx + yVec2 * alphaVecy;
						zVec3 += xVec3 * alphaVecx + yVec3 * alphaVecy;
					}
					xHead = ref Unsafe.Add(ref xHead, iterSize);
					xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
					xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
					xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
					yHead = ref Unsafe.Add(ref yHead, iterSize);
					yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
					yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
					yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
					zVec0.StoreUnsafe(ref zHead);
					zVec1.StoreUnsafe(ref zHead1);
					zVec2.StoreUnsafe(ref zHead2);
					zVec3.StoreUnsafe(ref zHead3);
					zHead = ref Unsafe.Add(ref zHead, iterSize);
					zHead1 = ref Unsafe.Add(ref zHead1, iterSize);
					zHead2 = ref Unsafe.Add(ref zHead2, iterSize);
					zHead3 = ref Unsafe.Add(ref zHead3, iterSize);
				}
			}
			for (; i < length; i++)
			{
				zHead += alphax * xHead;
				zHead += alphay * yHead;
				xHead = ref Unsafe.Add(ref xHead, 1);
				yHead = ref Unsafe.Add(ref yHead, 1);
				zHead = ref Unsafe.Add(ref zHead, 1);
			}
		}

		public static void Axpy2V_Kernel(ref scalar xHead, scalar alphax, scalar alphay, ref scalar yHead, ref scalar zHead, TripleIndice indice)
		{
			for (int i = 0; i < indice.Length; i++)
			{
				scalar sum = alphax * xHead + alphay * yHead;
				zHead += sum;
				xHead = ref Unsafe.Add(ref xHead, indice.AStride);
				yHead = ref Unsafe.Add(ref yHead, indice.BStride);
				zHead = ref Unsafe.Add(ref zHead, indice.CStride);
			}
		}
	}

	public static void DotAxpy(scalar alpha, 
		in vector x, in vector y, ref scalar rho, in vector z)
	{
		TripleIndice indice = CheckIndice(x, y, z);
		Details.DotAxpyV_Impl(ref x[0], alpha, ref y[0], ref rho, ref z[0], indice);
	}

	public static partial class Details
	{
		public static void DotAxpyV_Impl(ref scalar xHead, scalar alpha, ref scalar yHead,
            ref scalar rho, ref scalar zHead, TripleIndice indice)
		{
			if (indice.Length == 0)
				rho = 0.0;
			else if (alpha == 0.0)
				DotV_Impl(ref xHead, ref yHead, indice.AB, out rho);
			else if ((indice.AStride == 1) && (indice.BStride == 1) && (indice.CStride == 1))
				DotAxpyV_Kernel_Vector256(indice.Length, alpha,
					ref xHead, ref yHead, ref rho, ref zHead);
			else
				DotAxpyV_Kernel(ref xHead, alpha, ref yHead, ref rho, ref zHead, indice);
		}

		private static void DotAxpyV_Kernel_Vector256(int length, scalar alpha, ref scalar xHead, ref scalar yHead, ref scalar rho, ref scalar zHead)
		{
			rho = 0.0;
			int iterStride = 3;
			int iterSize = iterStride * Vector256<scalar>.Count;
			int i = 0;
			if (length >= iterSize)
			{
				ref var xHead1 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count);
				ref var yHead1 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count);
				ref var zHead1 = ref Unsafe.Add(ref zHead, Vector256<scalar>.Count);
				ref var xHead2 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 2);
				ref var yHead2 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 2);
				ref var zHead2 = ref Unsafe.Add(ref zHead, Vector256<scalar>.Count * 2);
				Vector256<scalar> rhoVec0 = Vector256<scalar>.Zero;
				Vector256<scalar> rhoVec1 = Vector256<scalar>.Zero;
				Vector256<scalar> rhoVec2 = Vector256<scalar>.Zero;
				Vector256<scalar> alphaVec = Vector256.Create(alpha);
				for (; i <= length - iterSize; i += iterSize)
				{
					Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xHead);
					Vector256<scalar> yVec0 = Vector256.LoadUnsafe(ref yHead);
					Vector256<scalar> zVec0 = Vector256.LoadUnsafe(ref zHead);
					Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xHead1);
					Vector256<scalar> yVec1 = Vector256.LoadUnsafe(ref yHead1);
					Vector256<scalar> zVec1 = Vector256.LoadUnsafe(ref zHead1);
					Vector256<scalar> xVec2 = Vector256.LoadUnsafe(ref xHead2);
					Vector256<scalar> yVec2 = Vector256.LoadUnsafe(ref yHead2);
					Vector256<scalar> zVec2 = Vector256.LoadUnsafe(ref zHead2);
					if (Fma.IsSupported)
					{
						rhoVec0 = Fma.MultiplyAdd(xVec0, yVec0, rhoVec0);
						rhoVec1 = Fma.MultiplyAdd(xVec1, yVec1, rhoVec1);
						rhoVec2 = Fma.MultiplyAdd(xVec2, yVec2, rhoVec2);
						zVec0 = Fma.MultiplyAdd(xVec0, alphaVec, zVec0);
						zVec1 = Fma.MultiplyAdd(xVec1, alphaVec, zVec1);
						zVec2 = Fma.MultiplyAdd(xVec2, alphaVec, zVec2);
					}
					else
					{
						rhoVec0 += xVec0 * yVec0;
						rhoVec1 += xVec1 * yVec1;
						rhoVec2 += xVec2 * yVec2;
						zVec0 += xVec0 * alphaVec;
						zVec1 += xVec1 * alphaVec;
						zVec2 += xVec2 * alphaVec;
					}
					xHead = ref Unsafe.Add(ref xHead, iterSize);
					xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
					xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
					yHead = ref Unsafe.Add(ref yHead, iterSize);
					yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
					yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
					zVec0.StoreUnsafe(ref zHead);
					zVec1.StoreUnsafe(ref zHead1);
					zVec2.StoreUnsafe(ref zHead2);
					zHead = ref Unsafe.Add(ref zHead, iterSize);
					zHead1 = ref Unsafe.Add(ref zHead1, iterSize);
					zHead2 = ref Unsafe.Add(ref zHead2, iterSize);
				}
				rhoVec0 += rhoVec1;
				rhoVec0 += rhoVec2;
				rho += Vector256.Sum(rhoVec0);
			}
			for (; i < length; i++)
			{
				rho += xHead * yHead;
				zHead += alpha * xHead;
				xHead = ref Unsafe.Add(ref xHead, 1);
				yHead = ref Unsafe.Add(ref yHead, 1);
				zHead = ref Unsafe.Add(ref zHead, 1);
			}
		}

		public static void DotAxpyV_Kernel(ref scalar xHead, scalar alpha,
			ref scalar yHead, ref scalar rho, ref scalar zHead, TripleIndice indice)
		{
			rho = 0.0;
			for (int i = 0; i < indice.Length; i++)
			{
				rho += xHead * yHead;
				zHead += alpha * xHead;
				xHead = ref Unsafe.Add(ref xHead, indice.AStride);
				yHead = ref Unsafe.Add(ref yHead, indice.BStride);
				zHead = ref Unsafe.Add(ref zHead, indice.CStride);
			}
		}
	}

	public static void AxpyF(scalar alpha,
        in matrix a, in vector x, in vector y)
	{
		var rows = a.Rows;
		var cols = a.Cols;
		if (x.Length != cols || y.Length != rows)
			throw new ArgumentException("Matrix and vector dimensions do not match.");
		Details.AxpyF_Impl(rows, cols, alpha, ref a[0, 0], a.RowStride, a.ColStride, ref x[0], x.Stride, ref y[0], y.Stride);
	}

	public static partial class Details
	{
		public static void AxpyF_Impl(int rows, int cols, scalar alpha, ref scalar aHead, int aRowStride, int aColStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
		{
			if ((rows == 0) || (cols == 0))
				return;
			else if (cols == 1)
				UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>(ref aHead,
					alpha * xHead, ref yHead, new DoubleIndice(rows, xStride, yStride), default);
			else
			{
				if (aColStride == 1 && rows == AxpyF_Kernel_RowMajor_Vector256_PerferredCount)
					AxpyF_Kernel_RowMajor_Vector256_Perferred_4p8(rows, alpha, ref aHead, aRowStride, ref xHead, xStride, ref yHead, yStride);
				else if (aRowStride == 1 && cols == AxpyF_Kernel_ColMajor_Vector256_PerferredCount)
					AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(rows, alpha, ref aHead, aColStride, ref xHead, xStride, ref yHead, yStride);
				else
					AxpyF_Kernel(rows, cols, alpha, ref aHead, aRowStride, aColStride, ref xHead, xStride, ref yHead, yStride);
			}
		}

		public const int AxpyF_Kernel_RowMajor_Vector256_PerferredCount = 8;

		public static void AxpyF_Kernel_RowMajor_Vector256_Perferred_4p8(int length, scalar alpha, ref scalar aHead, int aRowStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
		{
			int iterSize = 4;
			int i = 0;
			const int pref = AxpyF_Kernel_RowMajor_Vector256_PerferredCount;
			Span<scalar> xBuffer = stackalloc scalar[(int)pref];
			Scal2V_Kernel_Vector256(pref, alpha, ref xHead, ref xBuffer[0]);
			ref var xBuffHead = ref xBuffer[0];
			using var yBuffer = new BufferDVectorSpan(ref yHead, AxpyF_Kernel_RowMajor_Vector256_PerferredCount, yStride, shouldCopyBack: true);
			yStride = 1;
			yHead = ref yBuffer.bufferHead;
			ref var aHead1 = ref Unsafe.Add(ref aHead, 4);

			Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xBuffHead);
			Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xBuffHead, 4);

			if (length >= iterSize)
			{
				ref var yHead1 = ref Unsafe.Add(ref yHead, 1);
				ref var yHead2 = ref Unsafe.Add(ref yHead, 2);
				ref var yHead3 = ref Unsafe.Add(ref yHead, 3);

				for (; i <= length - iterSize; i += iterSize)
				{
					Vector256<scalar> aVec00 = Vector256.LoadUnsafe(ref aHead);
					Vector256<scalar> aVec01 = Vector256.LoadUnsafe(ref aHead1);
					aHead = ref Unsafe.Add(ref aHead, aRowStride);
					aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
					Vector256<scalar> yVec0 = aVec00 * xVec0;
					yVec0 = Fma.MultiplyAdd(aVec01, xVec1, yVec0);

					Vector256<scalar> aVec10 = Vector256.LoadUnsafe(ref aHead);
					Vector256<scalar> aVec11 = Vector256.LoadUnsafe(ref aHead1);
					aHead = ref Unsafe.Add(ref aHead, aRowStride);
					aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
					Vector256<scalar> yVec1 = aVec10 * xVec0;
					yVec1 = Fma.MultiplyAdd(aVec11, xVec1, yVec0);

					Vector256<scalar> aVec20 = Vector256.LoadUnsafe(ref aHead);
					Vector256<scalar> aVec21 = Vector256.LoadUnsafe(ref aHead1);
					aHead = ref Unsafe.Add(ref aHead, aRowStride);
					aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
					Vector256<scalar> yVec2 = aVec20 * xVec0;
					yVec2 = Fma.MultiplyAdd(aVec21, xVec1, yVec0);

					Vector256<scalar> aVec30 = Vector256.LoadUnsafe(ref aHead);
					Vector256<scalar> aVec31 = Vector256.LoadUnsafe(ref aHead1);
					aHead = ref Unsafe.Add(ref aHead, aRowStride);
					aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
					Vector256<scalar> yVec3 = aVec30 * xVec0;
					yVec3 = Fma.MultiplyAdd(aVec31, xVec1, yVec0);

					yHead += Vector256.Sum(yVec0);
					yHead1 += Vector256.Sum(yVec1);
					yHead2 += Vector256.Sum(yVec2);
					yHead3 += Vector256.Sum(yVec3);

					yHead = ref Unsafe.Add(ref yHead, iterSize);
					yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
					yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
					yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
				}
			}

			for (; i < length; i++)
			{
				Vector256<scalar> aVec0 = Vector256.LoadUnsafe(ref aHead);
				Vector256<scalar> aVec1 = Vector256.LoadUnsafe(ref aHead1);
				aHead = ref Unsafe.Add(ref aHead, aRowStride);
				aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
				Vector256<scalar> yVec = aVec0 * xVec0;
				yVec = Fma.MultiplyAdd(aVec1, xVec1, yVec);
				yHead += Vector256.Sum(yVec);
				yHead = ref Unsafe.Add(ref yHead, 1);
			}
		}


		public const int AxpyF_Kernel_ColMajor_Vector256_PerferredCount = 4;

		public static void AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(int length, scalar alpha, ref scalar aHead, int aColStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
		{
			using var yBuffer = new BufferDVectorSpan(ref yHead, AxpyF_Kernel_ColMajor_Vector256_PerferredCount, yStride, shouldCopyBack: true);
			yStride = 1;
			yHead = ref yBuffer.bufferHead;
			int iterStride = 2;
			int iterSize = iterStride * Vector256<scalar>.Count;
			int i = 0;
			ref scalar aRef0 = ref aHead;
			ref scalar aRef1 = ref Unsafe.Add(ref aHead, aColStride);
			ref scalar aRef2 = ref Unsafe.Add(ref aHead, 2 * aColStride);
			ref scalar aRef3 = ref Unsafe.Add(ref aHead, 3 * aColStride);
			scalar fac0 = alpha * xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar fac1 = alpha * xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar fac2 = alpha * xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar fac3 = alpha * xHead;

			if (length >= iterSize)
			{
				Vector256<scalar> alpha0 = Vector256.Create(fac0);
				Vector256<scalar> alpha1 = Vector256.Create(fac1);
				Vector256<scalar> alpha2 = Vector256.Create(fac2);
				Vector256<scalar> alpha3 = Vector256.Create(fac3);
				ref scalar yHead1 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count);

				for (; i <= length - iterSize; i += iterSize)
				{
					Vector256<scalar> y0 = Vector256.LoadUnsafe(ref yHead);
					Vector256<scalar> y1 = Vector256.LoadUnsafe(ref yHead1);
					Vector256<scalar> aVec00 = Vector256.LoadUnsafe(ref aRef0);
					aRef0 = ref Unsafe.Add(ref aRef0, Vector256<scalar>.Count);
					Vector256<scalar> aVec10 = Vector256.LoadUnsafe(ref aRef0);
					aRef0 = ref Unsafe.Add(ref aRef0, Vector256<scalar>.Count);
					Vector256<scalar> aVec01 = Vector256.LoadUnsafe(ref aRef1);
					aRef1 = ref Unsafe.Add(ref aRef1, Vector256<scalar>.Count);
					Vector256<scalar> aVec11 = Vector256.LoadUnsafe(ref aRef1);
					aRef1 = ref Unsafe.Add(ref aRef1, Vector256<scalar>.Count);
					Vector256<scalar> aVec02 = Vector256.LoadUnsafe(ref aRef2);
					aRef2 = ref Unsafe.Add(ref aRef2, Vector256<scalar>.Count);
					Vector256<scalar> aVec12 = Vector256.LoadUnsafe(ref aRef2);
					aRef2 = ref Unsafe.Add(ref aRef2, Vector256<scalar>.Count);
					Vector256<scalar> aVec03 = Vector256.LoadUnsafe(ref aRef3);
					aRef3 = ref Unsafe.Add(ref aRef3, Vector256<scalar>.Count);
					Vector256<scalar> aVec13 = Vector256.LoadUnsafe(ref aRef3);
					aRef3 = ref Unsafe.Add(ref aRef3, Vector256<scalar>.Count);

					y0 = Fma.MultiplyAdd(aVec00, alpha0, y0);
					y1 = Fma.MultiplyAdd(aVec10, alpha0, y1);
					y0 = Fma.MultiplyAdd(aVec01, alpha1, y0);
					y1 = Fma.MultiplyAdd(aVec11, alpha1, y1);
					y0 = Fma.MultiplyAdd(aVec02, alpha2, y0);
					y1 = Fma.MultiplyAdd(aVec12, alpha2, y1);
					y0 = Fma.MultiplyAdd(aVec03, alpha3, y0);
					y1 = Fma.MultiplyAdd(aVec13, alpha3, y1);

					y0.StoreUnsafe(ref yHead);
					y1.StoreUnsafe(ref yHead1);
					yHead = ref Unsafe.Add(ref yHead, iterSize);
					yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
				}
			}

			for (; i < length; i++)
			{
				scalar yVal = yHead;
				yVal += fac0 * aRef0;
				yVal += fac1 * aRef1;
				yVal += fac2 * aRef2;
				yVal += fac3 * aRef3;
				yHead = yVal;

				yHead = ref Unsafe.Add(ref yHead, 1);
				aRef0 = ref Unsafe.Add(ref aRef0, 1);
				aRef1 = ref Unsafe.Add(ref aRef1, 1);
				aRef2 = ref Unsafe.Add(ref aRef2, 1);
				aRef3 = ref Unsafe.Add(ref aRef3, 1);
			}
		}

		public static void AxpyF_Kernel(int rows, int cols, scalar alpha, ref scalar aHead, int aRowStride, int aColStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
		{
			for (int i = 0; i < cols; i++)
			{
				UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>(ref aHead, alpha * xHead,
					ref yHead, new DoubleIndice(rows, aRowStride, yStride), default);
				aHead = ref Unsafe.Add(ref aHead, aColStride);
				xHead = ref Unsafe.Add(ref xHead, xStride);
			}
		}

		private static void Scal2V_Kernel_Vector256(int length, scalar alpha, ref scalar xHead, ref scalar yHead)
		{
			int iterStride = 4;
			int iterSize = iterStride * Vector256<scalar>.Count;
			int i = 0;
			if (length >= iterSize)
			{
				ref var xHead1 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count);
				ref var yHead1 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count);
				ref var xHead2 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 2);
				ref var yHead2 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 2);
				ref var xHead3 = ref Unsafe.Add(ref xHead, Vector256<scalar>.Count * 3);
				ref var yHead3 = ref Unsafe.Add(ref yHead, Vector256<scalar>.Count * 3);
				Vector256<scalar> alphaVec = Vector256.Create(alpha);
				for (; i <= length - iterSize; i += iterSize)
				{
					Vector256<scalar> xVec0 = Vector256.LoadUnsafe(ref xHead);
					Vector256<scalar> xVec1 = Vector256.LoadUnsafe(ref xHead1);
					Vector256<scalar> xVec2 = Vector256.LoadUnsafe(ref xHead2);
					Vector256<scalar> xVec3 = Vector256.LoadUnsafe(ref xHead3);
					xVec0 *= alphaVec;
					xVec1 *= alphaVec;
					xVec2 *= alphaVec;
					xVec3 *= alphaVec;
					xHead = ref Unsafe.Add(ref xHead, iterSize);
					xHead1 = ref Unsafe.Add(ref xHead1, iterSize);
					xHead2 = ref Unsafe.Add(ref xHead2, iterSize);
					xHead3 = ref Unsafe.Add(ref xHead3, iterSize);
					xVec0.StoreUnsafe(ref yHead);
					xVec1.StoreUnsafe(ref yHead1);
					xVec2.StoreUnsafe(ref yHead2);
					xVec3.StoreUnsafe(ref yHead3);
					yHead = ref Unsafe.Add(ref yHead, iterSize);
					yHead1 = ref Unsafe.Add(ref yHead1, iterSize);
					yHead2 = ref Unsafe.Add(ref yHead2, iterSize);
					yHead3 = ref Unsafe.Add(ref yHead3, iterSize);
				}
			}
			for (; i < length; i++)
			{
				yHead = alpha * xHead;
				xHead = ref Unsafe.Add(ref xHead, 1);
				yHead = ref Unsafe.Add(ref yHead, 1);
			}
		}
	}

	public static void DotxF(scalar alpha,
        in matrix a, in vector x, scalar beta, in vector y)
	{
		if (alpha == 0)
		{
			Scal(beta, y);
			return;
		}
		var rows = a.Rows;
		var cols = a.Cols;
		if (y.Length != cols || x.Length != rows)
			throw new ArgumentException("Matrix and vector dimensions do not match.");
		Details.DotxF_Impl(rows, cols, alpha, ref a[0, 0], a.RowStride, a.ColStride, ref x[0], x.Stride, beta, ref y[0], y.Stride);
	}

	public static partial class Details
	{
		public static void DotxF_Impl(int rows, int cols, scalar alpha, ref scalar aHead, int aRowStride, int aColStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
		{
			if ((rows == 0) || (cols == 0))
				return;
			else if (cols == 1)
			{
				yHead *= beta;
				DotV_Impl(ref aHead, ref xHead, new(rows, aRowStride, xStride), out var rho);
				yHead += alpha * rho;
			}
			else if (aColStride == 1 && cols == DotxF_Kernel_RowMajor_Vector256_PerferredCount)
				DotxF_Kernel_RowMajor_Vector256_Perferred_4p8(rows, alpha, ref aHead, aRowStride, ref xHead, xStride, beta, ref yHead, yStride);
			else if (aRowStride == 1 && cols == DotxF_Kernel_ColMajor_Vector256_PerferredCount)
				DotxF_Kernel_ColMajor_Vector256_Perferred_4p6(rows, alpha, ref aHead, aColStride, ref xHead, xStride, beta, ref yHead, yStride);
			else
				DotxF_Kernel(rows, cols, alpha, ref aHead, aRowStride, aColStride, ref xHead, xStride, beta, ref yHead, yStride);
		}

		public const int DotxF_Kernel_RowMajor_Vector256_PerferredCount = 8;

		public static void DotxF_Kernel_RowMajor_Vector256_Perferred_4p8(int rows, scalar alpha, ref scalar aHead, int aRowStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
		{
			int iterSize = 4;
			Span<scalar> yBuffer = stackalloc scalar[(int)DotxF_Kernel_RowMajor_Vector256_PerferredCount];
			ref var yBufferHead = ref MemoryMarshal.GetReference(yBuffer);
			Vector256<scalar> yVec0 = Vector256.LoadUnsafe(ref yBufferHead);
			Vector256<scalar> yVec1 = Vector256.LoadUnsafe(ref yBufferHead, 4);

			ref var aHead1 = ref Unsafe.Add(ref aHead, 4);
			int i = 0;
			for (; i <= rows - iterSize; i += iterSize)
			{
				scalar xScalar0 = xHead * alpha;
				Vector256<scalar> xVec0 = Vector256.Create(xScalar0);
				Vector256<scalar> aVec00 = Vector256.LoadUnsafe(ref aHead);
				Vector256<scalar> aVec10 = Vector256.LoadUnsafe(ref aHead1);
				xHead = ref Unsafe.Add(ref xHead, xStride);
				aHead = ref Unsafe.Add(ref aHead, aRowStride);
				aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
				yVec0 = Fma.MultiplyAdd(aVec00, xVec0, yVec0);
				yVec1 = Fma.MultiplyAdd(aVec10, xVec0, yVec1);

				scalar xScalar1 = xHead * alpha;
				Vector256<scalar> xVec1 = Vector256.Create(xScalar1);
				Vector256<scalar> aVec01 = Vector256.LoadUnsafe(ref aHead);
				Vector256<scalar> aVec11 = Vector256.LoadUnsafe(ref aHead1);
				xHead = ref Unsafe.Add(ref xHead, xStride);
				aHead = ref Unsafe.Add(ref aHead, aRowStride);
				aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
				yVec0 = Fma.MultiplyAdd(aVec01, xVec1, yVec0);
				yVec1 = Fma.MultiplyAdd(aVec11, xVec1, yVec1);

				scalar xScalar2 = xHead * alpha;
				Vector256<scalar> xVec2 = Vector256.Create(xScalar2);
				Vector256<scalar> aVec02 = Vector256.LoadUnsafe(ref aHead);
				Vector256<scalar> aVec12 = Vector256.LoadUnsafe(ref aHead1);
				xHead = ref Unsafe.Add(ref xHead, xStride);
				aHead = ref Unsafe.Add(ref aHead, aRowStride);
				aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
				yVec0 = Fma.MultiplyAdd(aVec02, xVec2, yVec0);
				yVec1 = Fma.MultiplyAdd(aVec12, xVec2, yVec1);

				scalar xScalar3 = xHead * alpha;
				Vector256<scalar> xVec3 = Vector256.Create(xScalar3);
				Vector256<scalar> aVec03 = Vector256.LoadUnsafe(ref aHead);
				Vector256<scalar> aVec13 = Vector256.LoadUnsafe(ref aHead1);
				xHead = ref Unsafe.Add(ref xHead, xStride);
				aHead = ref Unsafe.Add(ref aHead, aRowStride);
				aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
				yVec0 = Fma.MultiplyAdd(aVec03, xVec3, yVec0);
				yVec1 = Fma.MultiplyAdd(aVec13, xVec3, yVec1);
			}
			for (; i < rows; i++)
			{
				scalar xScalar = xHead * alpha;
				Vector256<scalar> xVec = Vector256.Create(xScalar);
				Vector256<scalar> aVec0 = Vector256.LoadUnsafe(ref aHead);
				Vector256<scalar> aVec1 = Vector256.LoadUnsafe(ref aHead1);
				xHead = ref Unsafe.Add(ref xHead, xStride);
				aHead = ref Unsafe.Add(ref aHead, aRowStride);
				aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
				yVec0 = Fma.MultiplyAdd(aVec0, xVec, yVec0);
				yVec1 = Fma.MultiplyAdd(aVec1, xVec, yVec1);
			}

			yVec0.StoreUnsafe(ref yBufferHead);
			yVec1.StoreUnsafe(ref yBufferHead, 4);
			for (i = 0; i < DotxF_Kernel_RowMajor_Vector256_PerferredCount; i++)
			{
				yHead *= beta;
				yHead += yBufferHead;
				yHead = ref Unsafe.Add(ref yHead, yStride);
				yBufferHead = ref Unsafe.Add(ref yBufferHead, 1);
			}
		}

		public const int DotxF_Kernel_ColMajor_Vector256_PerferredCount = 6;

		public static void DotxF_Kernel_ColMajor_Vector256_Perferred_4p6(int rows, scalar alpha, 
			ref scalar aHead, int aColStride, ref scalar xHead, int xStride, scalar beta, 
			ref scalar yHead, int yStride)
		{
			using var xBuffer = new BufferDVectorSpan(ref xHead, rows, xStride, alpha);
			xHead = ref xBuffer.bufferHead;
			xStride = 1;
			UFunc.Details.Map_Kernel<MultiplyOperator<scalar>, scalar>(ref yHead, beta, 
				new(DotxF_Kernel_ColMajor_Vector256_PerferredCount, yStride), new());
			Span<scalar> vals = stackalloc scalar[(int)DotxF_Kernel_ColMajor_Vector256_PerferredCount];
			ref scalar aRef0 = ref aHead;
			ref scalar aRef1 = ref Unsafe.Add(ref aHead, aColStride);
			ref scalar aRef2 = ref Unsafe.Add(ref aHead, 2 * aColStride);
			ref scalar aRef3 = ref Unsafe.Add(ref aHead, 3 * aColStride);
			ref scalar aRef4 = ref Unsafe.Add(ref aHead, 4 * aColStride);
			ref scalar aRef5 = ref Unsafe.Add(ref aHead, 5 * aColStride);

			int i = 0;
			int iterStride = 1;
			int iterSize = iterStride * Vector256<scalar>.Count;
			if (rows >= iterSize)
			{

				Vector256<scalar> yVec0 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec1 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec2 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec3 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec4 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec5 = Vector256<scalar>.Zero;

				for (; i <= rows - iterSize; i += iterSize)
				{
					Vector256<scalar> aVec0 = Vector256.LoadUnsafe(ref aRef0);
					Vector256<scalar> aVec1 = Vector256.LoadUnsafe(ref aRef1);
					Vector256<scalar> aVec2 = Vector256.LoadUnsafe(ref aRef2);
					Vector256<scalar> aVec3 = Vector256.LoadUnsafe(ref aRef3);
					Vector256<scalar> aVec4 = Vector256.LoadUnsafe(ref aRef4);
					Vector256<scalar> aVec5 = Vector256.LoadUnsafe(ref aRef5);

					Vector256<scalar> xVec = Vector256.LoadUnsafe(ref xHead);

					yVec0 = Fma.MultiplyAdd(aVec0, xVec, yVec0);
					yVec1 = Fma.MultiplyAdd(aVec1, xVec, yVec1);
					yVec2 = Fma.MultiplyAdd(aVec2, xVec, yVec2);
					yVec3 = Fma.MultiplyAdd(aVec3, xVec, yVec3);
					yVec4 = Fma.MultiplyAdd(aVec4, xVec, yVec4);
					yVec5 = Fma.MultiplyAdd(aVec5, xVec, yVec5);

					aRef0 = ref Unsafe.Add(ref aRef0, iterSize);
					aRef1 = ref Unsafe.Add(ref aRef1, iterSize);
					aRef2 = ref Unsafe.Add(ref aRef2, iterSize);
					aRef3 = ref Unsafe.Add(ref aRef3, iterSize);
					aRef4 = ref Unsafe.Add(ref aRef4, iterSize);
					aRef5 = ref Unsafe.Add(ref aRef5, iterSize);
					xHead = ref Unsafe.Add(ref xHead, iterSize);
				}

				vals[0] = Vector256.Sum(yVec0);
				vals[1] = Vector256.Sum(yVec1);
				vals[2] = Vector256.Sum(yVec2);
				vals[3] = Vector256.Sum(yVec3);
				vals[4] = Vector256.Sum(yVec4);
				vals[5] = Vector256.Sum(yVec5);
			}
			for (; i < rows; i++)
			{
				scalar val = xHead;
				vals[0] += aRef0 * val;
				vals[1] += aRef1 * val;
				vals[2] += aRef2 * val;
				vals[3] += aRef3 * val;
				vals[4] += aRef4 * val;
				vals[5] += aRef5 * val;

				aRef0 = ref Unsafe.Add(ref aRef0, 1);
				aRef1 = ref Unsafe.Add(ref aRef1, 1);
				aRef2 = ref Unsafe.Add(ref aRef2, 1);
				aRef3 = ref Unsafe.Add(ref aRef3, 1);
				aRef4 = ref Unsafe.Add(ref aRef4, 1);
				aRef5 = ref Unsafe.Add(ref aRef5, 1);
				xHead = ref Unsafe.Add(ref xHead, 1);
			}
			UFunc.Details.Combine_Kernel<AddOperator<scalar>>(ref vals[0], ref yHead,
			new(DotxF_Kernel_ColMajor_Vector256_PerferredCount, 1, yStride), default);
		}

		public static void DotxF_Kernel(int rows, int cols, scalar alpha, ref scalar aHead, int aRowStride, int aColStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
		{
			UFunc.Details.Map_Kernel<MultiplyOperator<scalar>, scalar>
				(ref yHead, beta, new(cols, yStride), new());
			ref scalar aColHead = ref aHead;
			for (int i = 0; i < cols; i++)
			{
				DotV_Kernel(ref aColHead, ref xHead, new(rows, aRowStride, xStride), out var rho);
				aColHead = ref Unsafe.Add(ref aColHead, aColStride);
				yHead += alpha * rho;
				yHead = ref Unsafe.Add(ref yHead, yStride);
			}
		}
	}

	public static void DotxAxpyF(scalar alpha, 
		in matrix a, in vector w, in vector x,
		scalar beta, in vector y, in vector z)
	{
		var rows = a.Rows;
		var cols = a.Cols;
		if (z.Length != rows || w.Length != rows || x.Length != cols || y.Length != cols)
			throw new ArgumentException("Matrix and vector dimensions do not match.");
		Details.DotxAxpyF_Impl(rows, cols,
					alpha,
					ref a[0, 0], a.RowStride, a.ColStride,
					ref w[0], w.Stride,
					ref x[0], x.Stride,
					beta,
					ref y[0], y.Stride,
					ref z[0], z.Stride);
	}

	public static partial class Details
	{
		public static void DotxAxpyF_Impl(int rows, int cols,
			 scalar alpha,
			ref scalar aHead, int aRowStride, int aColStride,
			ref scalar wHead, int wStride,
			ref scalar xHead, int xStride,
			 scalar beta,
			ref scalar yHead, int yStride,
			ref scalar zHead, int zStride)
		{
			if ((rows == 0) || (cols == 0))
				return;
			else
			{
				if (cols == 1)
				{
					DotV_Impl(ref aHead, ref xHead, new(rows, aRowStride, wStride), out var rho);
					yHead = beta * yHead + alpha * rho;
					UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>(ref aHead,
						alpha * xHead, ref zHead, new DoubleIndice(rows, aRowStride, zStride), default);
				}
				else if (aColStride == 1 && cols == DotxAxpyF_Kernel_RowMajor_Vector256_PerferredCount)
					DotxAxpyF_Kernel_RowMajor_Vector256_Perferred_4p4(rows,
						alpha,
						ref aHead, aRowStride,
						ref wHead, wStride,
						ref xHead, xStride,
						beta,
						ref yHead, yStride,
						ref zHead, zStride);
				else if (aRowStride == 1 && cols == DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount)
					DotxAxpyF_Kernel_ColMajor_Vector256_Perferred_4p4(rows,
						alpha,
						ref aHead, aColStride,
						ref wHead, wStride,
						ref xHead, xStride,
						beta,
						ref yHead, yStride,
						ref zHead, zStride);
				else
					DotxAxpyF_Kernel(rows, cols,
						alpha,
						ref aHead, aRowStride, aColStride,
						ref wHead, wStride,
						ref xHead, xStride,
						beta,
						ref yHead, yStride,
						ref zHead, zStride);
			}
		}

		public const int DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount = 4;

		public static void DotxAxpyF_Kernel_ColMajor_Vector256_Perferred_4p4(int rows,
			scalar alpha,
			ref scalar aHead, int aColStride,
			ref scalar wHead, int wStride,
			ref scalar xHead, int xStride,
			scalar beta,
			ref scalar yHead, int yStride,
			ref scalar zHead, int zStride)
		{
			int pref = DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount;
			int iterStride = 1;
			int iterSize = iterStride * Vector256<scalar>.Count;
			int i = 0;

			scalar x0 = xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar x1 = xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar x2 = xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar x3 = xHead;

			Span<scalar> yBuffer = stackalloc scalar[(int)pref];
			ref var yBufferHead = ref MemoryMarshal.GetReference(yBuffer);

			using var wBuffer = new BufferDVectorSpan(ref wHead, rows, wStride, shouldCopyBack: false);
			wHead = ref wBuffer.bufferHead;
			wStride = 1;

			using var zBuffer = new BufferDVectorSpan(ref zHead, rows, zStride, shouldCopyBack: true);
			zHead = ref zBuffer.bufferHead;
			zStride = 1;

			i = 0;
			ref scalar aHead1 = ref Unsafe.Add(ref aHead, aColStride);
			ref scalar aHead2 = ref Unsafe.Add(ref aHead, 2 * aColStride);
			ref scalar aHead3 = ref Unsafe.Add(ref aHead, 3 * aColStride);
			Vector256<scalar> alphaVec = Vector256.Create(alpha);

			if (rows >= iterSize)
			{

				Vector256<scalar> xVec0 = Vector256.Create(x0);
				Vector256<scalar> xVec1 = Vector256.Create(x1);
				Vector256<scalar> xVec2 = Vector256.Create(x2);
				Vector256<scalar> xVec3 = Vector256.Create(x3);
				Vector256<scalar> yVec0 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec1 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec2 = Vector256<scalar>.Zero;
				Vector256<scalar> yVec3 = Vector256<scalar>.Zero;

				for (; i <= rows - iterSize; i += iterSize)
				{
					Vector256<scalar> aVec0 = Vector256.LoadUnsafe(ref aHead);
					aVec0 *= alphaVec;
					Vector256<scalar> aVec1 = Vector256.LoadUnsafe(ref aHead1);
					aVec1 *= alphaVec;
					Vector256<scalar> aVec2 = Vector256.LoadUnsafe(ref aHead2);
					aVec2 *= alphaVec;
					Vector256<scalar> aVec3 = Vector256.LoadUnsafe(ref aHead3);
					aVec3 *= alphaVec;

					Vector256<scalar> wVec = Vector256.LoadUnsafe(ref wHead);

					yVec0 = Fma.MultiplyAdd(aVec0, wVec, yVec0);
					yVec1 = Fma.MultiplyAdd(aVec1, wVec, yVec1);
					yVec2 = Fma.MultiplyAdd(aVec2, wVec, yVec2);
					yVec3 = Fma.MultiplyAdd(aVec3, wVec, yVec3);

					Vector256<scalar> zVec = Vector256.LoadUnsafe(ref zHead);
					zVec = Fma.MultiplyAdd(aVec0, xVec0, zVec);
					zVec = Fma.MultiplyAdd(aVec1, xVec1, zVec);
					zVec = Fma.MultiplyAdd(aVec2, xVec2, zVec);
					zVec = Fma.MultiplyAdd(aVec3, xVec3, zVec);

					zVec.StoreUnsafe(ref zHead);

					aHead = ref Unsafe.Add(ref aHead, iterSize);
					aHead1 = ref Unsafe.Add(ref aHead1, iterSize);
					aHead2 = ref Unsafe.Add(ref aHead2, iterSize);
					aHead3 = ref Unsafe.Add(ref aHead3, iterSize);
					wHead = ref Unsafe.Add(ref wHead, iterSize);
					zHead = ref Unsafe.Add(ref zHead, iterSize);
				}

				yBufferHead = Vector256.Sum(yVec0);
				yBufferHead = ref Unsafe.Add(ref yBufferHead, 1);
				yBufferHead = Vector256.Sum(yVec1);
				yBufferHead = ref Unsafe.Add(ref yBufferHead, 1);
				yBufferHead = Vector256.Sum(yVec2);
				yBufferHead = ref Unsafe.Add(ref yBufferHead, 1);
				yBufferHead = Vector256.Sum(yVec3);
				yBufferHead = ref MemoryMarshal.GetReference(yBuffer);
			}
			if (i < rows)
			{
				Vector256<scalar> yVec = Vector256.LoadUnsafe(ref yBufferHead);
				Vector256<scalar> xVec = Vector256.Create(x0, x1, x2, x3);
				for (; i < rows; i++)
				{
					Vector256<scalar> aVec = Vector256.Create(aHead, aHead1, aHead2, aHead3);
					Vector256<scalar> wVec = Vector256.Create(wHead);
					aVec *= alphaVec;
					yVec = Fma.MultiplyAdd(aVec, wVec, yVec);
					Vector256<scalar> zVec = aVec * xVec;
					zHead += Vector256.Sum(zVec);

					aHead = ref Unsafe.Add(ref aHead, 1);
					aHead1 = ref Unsafe.Add(ref aHead1, 1);
					aHead2 = ref Unsafe.Add(ref aHead2, 1);
					aHead3 = ref Unsafe.Add(ref aHead3, 1);
					wHead = ref Unsafe.Add(ref wHead, 1);
					zHead = ref Unsafe.Add(ref zHead, 1);
				}
				yVec.StoreUnsafe(ref yBufferHead);
			}
			for (i = 0; i < DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount; i++)
			{
				yHead = yBufferHead + beta * yHead;
				yHead = ref Unsafe.Add(ref yHead, yStride);
				yBufferHead = ref Unsafe.Add(ref yBufferHead, 1);
			}
		}

		public const int DotxAxpyF_Kernel_RowMajor_Vector256_PerferredCount = 4;

		public static void DotxAxpyF_Kernel_RowMajor_Vector256_Perferred_4p4(int rows,
			scalar alpha,
			ref scalar aHead, int aRowStride,
			ref scalar wHead, int wStride,
			ref scalar xHead, int xStride,
			scalar beta,
			ref scalar yHead, int yStride,
			ref scalar zHead, int zStride)
		{
			int pref = DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount;
			int iterStride = 1;
			int iterSize = iterStride * Vector256<scalar>.Count;
			int i = 0;

			Span<scalar> yBuffer = stackalloc scalar[(int)pref];
			ref var yBufferHead = ref MemoryMarshal.GetReference(yBuffer);
			Vector256<scalar> yVec = Vector256<scalar>.Zero;
			Vector256<scalar> xVec = Vector256.LoadUnsafe(ref xHead);

			scalar x0 = xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar x1 = xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar x2 = xHead;
			xHead = ref Unsafe.Add(ref xHead, xStride);
			scalar x3 = xHead;

			using var zBuffer = new BufferDVectorSpan(ref zHead, rows, zStride, shouldCopyBack: true);
			zHead = ref zBuffer.bufferHead;
			zStride = 1;

			i = 0;
			ref scalar aHead1 = ref Unsafe.Add(ref aHead, aRowStride);
			ref scalar aHead2 = ref Unsafe.Add(ref aHead, 2 * aRowStride);
			ref scalar aHead3 = ref Unsafe.Add(ref aHead, 3 * aRowStride);
			Vector256<scalar> alphaVec = Vector256.Create(alpha);

			if (rows >= iterSize)
			{

				ref var zHead1 = ref Unsafe.Add(ref zHead, 1);
				ref var zHead2 = ref Unsafe.Add(ref zHead, 2);
				ref var zHead3 = ref Unsafe.Add(ref zHead, 3);

				for (; i <= rows - iterSize; i += iterSize)
				{
					Vector256<scalar> aVec0 = Vector256.LoadUnsafe(ref aHead);
					aVec0 *= alphaVec;
					Vector256<scalar> aVec1 = Vector256.LoadUnsafe(ref aHead1);
					aVec1 *= alphaVec;
					Vector256<scalar> aVec2 = Vector256.LoadUnsafe(ref aHead2);
					aVec2 *= alphaVec;
					Vector256<scalar> aVec3 = Vector256.LoadUnsafe(ref aHead3);
					aVec3 *= alphaVec;

					Vector256<scalar> wVec0 = Vector256.Create(wHead);
					wHead = ref Unsafe.Add(ref wHead, wStride);
					Vector256<scalar> wVec1 = Vector256.Create(wHead);
					wHead = ref Unsafe.Add(ref wHead, wStride);
					Vector256<scalar> wVec2 = Vector256.Create(wHead);
					wHead = ref Unsafe.Add(ref wHead, wStride);
					Vector256<scalar> wVec3 = Vector256.Create(wHead);
					wHead = ref Unsafe.Add(ref wHead, wStride);

					yVec = Fma.MultiplyAdd(aVec0, wVec0, yVec);
					yVec = Fma.MultiplyAdd(aVec1, wVec1, yVec);
					yVec = Fma.MultiplyAdd(aVec2, wVec2, yVec);
					yVec = Fma.MultiplyAdd(aVec3, wVec3, yVec);

					Vector256<scalar> zVec0 = aVec0 * xVec;
					Vector256<scalar> zVec1 = aVec1 * xVec;
					Vector256<scalar> zVec2 = aVec2 * xVec;
					Vector256<scalar> zVec3 = aVec3 * xVec;
					zHead += Vector256.Sum(zVec0);
					zHead1 += Vector256.Sum(zVec1);
					zHead2 += Vector256.Sum(zVec2);
					zHead3 += Vector256.Sum(zVec3);

					aHead = ref Unsafe.Add(ref aHead, iterSize * aRowStride);
					aHead1 = ref Unsafe.Add(ref aHead1, iterSize * aRowStride);
					aHead2 = ref Unsafe.Add(ref aHead2, iterSize * aRowStride);
					aHead3 = ref Unsafe.Add(ref aHead3, iterSize * aRowStride);
					zHead = ref Unsafe.Add(ref zHead, iterSize);
					zHead1 = ref Unsafe.Add(ref zHead1, iterSize);
					zHead2 = ref Unsafe.Add(ref zHead2, iterSize);
					zHead3 = ref Unsafe.Add(ref zHead3, iterSize);
				}
			}
			if (i < rows)
			{
				for (; i < rows; i++)
				{
					Vector256<scalar> aVec = Vector256.LoadUnsafe(ref aHead);
					Vector256<scalar> wVec = Vector256.Create(wHead);
					aVec *= alphaVec;
					yVec = Fma.MultiplyAdd(aVec, wVec, yVec);
					Vector256<scalar> zVec = aVec * xVec;
					zHead += Vector256.Sum(zVec);

					aHead = ref Unsafe.Add(ref aHead, aRowStride);
					aHead1 = ref Unsafe.Add(ref aHead1, aRowStride);
					aHead2 = ref Unsafe.Add(ref aHead2, aRowStride);
					aHead3 = ref Unsafe.Add(ref aHead3, aRowStride);
					wHead = ref Unsafe.Add(ref wHead, 1);
					zHead = ref Unsafe.Add(ref zHead, 1);
				}
			}
			yVec.StoreUnsafe(ref yBufferHead);
			for (i = 0; i < DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount; i++)
			{
				yHead = yBufferHead + beta * yHead;
				yHead = ref Unsafe.Add(ref yHead, yStride);
				yBufferHead = ref Unsafe.Add(ref yBufferHead, 1);
			}
		}

		public static void DotxAxpyF_Kernel(int rows, int cols,
			 scalar alpha,
			ref scalar aHead, int aRowStride, int aColStride,
			ref scalar wHead, int wStride,
			ref scalar xHead, int xStride,
			 scalar beta,
			ref scalar yHead, int yStride,
			ref scalar zHead, int zStride)
		{
            MultiplyAddOperator<scalar> action = default;
            for (int i = 0; i < cols; i++)
			{
				DotV_Impl(ref aHead, ref wHead, new(rows, aRowStride, wStride), out var rho);
				UFunc.Details.Combine_Impl<MultiplyAddOperator<scalar>, scalar>(ref aHead,
					alpha * xHead, ref zHead, new DoubleIndice(rows, aRowStride, zStride), action);
				aHead = ref Unsafe.Add(ref aHead, aColStride);
				xHead = ref Unsafe.Add(ref xHead, xStride);
				yHead *= beta;
				yHead += alpha * rho;
				yHead = ref Unsafe.Add(ref yHead, yStride);
			}
		}
	}
}