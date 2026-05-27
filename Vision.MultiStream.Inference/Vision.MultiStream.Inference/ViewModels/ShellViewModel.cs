namespace Vision.MultiStream.Inference.ViewModels
{
    public class ShellViewModel
    {
        public ShellViewModel(
            SnapshotViewModel snapshot,
            MultiStreamViewModel multiStream,
            PerformanceViewModel performance)
        {
            Snapshot = snapshot;
            MultiStream = multiStream;
            Performance = performance;
        }

        public SnapshotViewModel Snapshot { get; }

        public MultiStreamViewModel MultiStream { get; }

        public PerformanceViewModel Performance { get; }
    }
}
