namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 탭 두 개(Snapshot / MultiStream)를 담는 최상위 ViewModel.
    /// </summary>
    public class ShellViewModel
    {
        public ShellViewModel(SnapshotViewModel snapshot, MultiStreamViewModel multiStream)
        {
            Snapshot = snapshot;
            MultiStream = multiStream;
        }

        public SnapshotViewModel Snapshot { get; }

        public MultiStreamViewModel MultiStream { get; }
    }
}
