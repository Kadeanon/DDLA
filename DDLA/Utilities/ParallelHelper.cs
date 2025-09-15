using System.Runtime.CompilerServices;

namespace DDLA.Utilities;

internal static class ParallelHelper
{/// <summary>
 /// Executes a specified action in an optimized parallel loop.
 /// </summary>
 /// <typeparam name="TAction">The type of action (implementing <see cref="IAction"/>) to invoke for each iteration index.</typeparam>
 /// <param name="start">The starting iteration index.</param>
 /// <param name="end">The final iteration index (exclusive).</param>
 /// <param name="action">The <typeparamref name="TAction"/> instance representing the action to invoke.</param>
 /// <param name="minimumActionsPerThread">
 /// The minimum number of actions to run per individual thread. Set to 1 if all invocations
 /// should be parallelized, or to a greater number if each individual invocation is fast
 /// enough that it is more efficient to set a lower bound per each running thread.
 /// </param>
    internal static void For<TAction>(int start, int end, in TAction action, int minimumActionsPerThread, int maxDegreeOfParallelism)
        where TAction : struct, IAction
    {

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minimumActionsPerThread,
            nameof(minimumActionsPerThread));

        ArgumentOutOfRangeException.ThrowIfGreaterThan(start, end,
            nameof(start));


        if (start == end)
        {
            return;
        }

        int count = Math.Abs(start - end);
        int maxBatches = 1 + (count - 1) / minimumActionsPerThread;
        int cores = Environment.ProcessorCount;
        int numBatches = Math.Min(maxBatches, cores);

        // Skip the parallel invocation when a single batch is needed
        if (numBatches == 1 || maxDegreeOfParallelism == 1)
        {
            for (int i = start; i < end; i++)
            {
                Unsafe.AsRef(in action).Invoke(i);
            }

            return;
        }

        int batchSize = 1 + (count - 1) / numBatches;

        ActionInvoker<TAction> actionInvoker = new(start, end, batchSize, action);
        int degreeOfParallelism = Math.Min(numBatches, maxDegreeOfParallelism);
        // Run the batched operations in parallel
        _ = Parallel.For(
            0,
            numBatches,
            new ParallelOptions { MaxDegreeOfParallelism = degreeOfParallelism },
            actionInvoker.Invoke);
    }

    // Wrapping struct acting as explicit closure to execute the processing batches
    private readonly struct ActionInvoker<TAction>
        where TAction : struct, IAction
    {
        private readonly int start;
        private readonly int end;
        private readonly int batchSize;
        private readonly TAction action;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ActionInvoker(
            int start,
            int end,
            int batchSize,
            in TAction action)
        {
            this.start = start;
            this.end = end;
            this.batchSize = batchSize;
            this.action = action;
        }

        /// <summary>
        /// Processes the batch of actions at a specified index
        /// </summary>
        /// <param name="i">The index of the batch to process</param>
        public void Invoke(int i)
        {
            int offset = i * batchSize;
            int low = start + offset;
            int high = low + batchSize;
            int stop = Math.Min(high, end);

            for (int j = low; j < stop; j++)
            {
                Unsafe.AsRef(in action).Invoke(j);
            }
        }
    }
}

/// <summary>
/// A contract for actions being executed with an input index.
/// </summary>
/// <remarks>If the <see cref="Invoke"/> method is small enough, it is highly recommended to mark it with <see cref="MethodImplOptions.AggressiveInlining"/>.</remarks>
public interface IAction
{
    /// <summary>
    /// Executes the action associated with a specific index.
    /// </summary>
    /// <param name="i">The current index for the action to execute.</param>
    void Invoke(int i);
}
