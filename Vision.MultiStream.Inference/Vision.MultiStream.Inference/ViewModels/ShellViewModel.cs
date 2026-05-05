namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 탭 두 개(Snapshot / Rtsp)를 담는 최상위 ViewModel.
    /// 자기 자신은 상태 없음 — 자식 VM 두 개를 그대로 노출.
    /// </summary>
    public class ShellViewModel
    {
        public ShellViewModel(SnapshotViewModel snapshot, RtspViewModel rtsp)
        {
            Snapshot = snapshot;
            Rtsp = rtsp;
        }

        public SnapshotViewModel Snapshot { get; }

        public RtspViewModel Rtsp { get; }
    }
}
