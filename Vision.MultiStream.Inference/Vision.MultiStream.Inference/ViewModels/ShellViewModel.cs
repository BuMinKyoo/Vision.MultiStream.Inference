namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 탭 세 개(Snapshot / MultiStream / Mp3Player)를 담는 최상위 ViewModel.
    /// </summary>
    public class ShellViewModel
    {
        public ShellViewModel(SnapshotViewModel snapshot, MultiStreamViewModel multiStream, Mp3PlayerViewModel mp3Player)
        {
            Snapshot = snapshot;
            MultiStream = multiStream;
            Mp3Player = mp3Player;
        }

        public SnapshotViewModel Snapshot { get; }

        public MultiStreamViewModel MultiStream { get; }

        public Mp3PlayerViewModel Mp3Player { get; }
    }
}
