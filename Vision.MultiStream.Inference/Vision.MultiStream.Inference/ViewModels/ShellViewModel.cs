namespace Vision.MultiStream.Inference.ViewModels
{
    public class ShellViewModel
    {
        public ShellViewModel(
            MultiStreamViewModel multiStream,
            PerformanceViewModel performance)
        {
            MultiStream = multiStream;
            Performance = performance;
        }

        public MultiStreamViewModel MultiStream { get; }

        public PerformanceViewModel Performance { get; }
    }
}
