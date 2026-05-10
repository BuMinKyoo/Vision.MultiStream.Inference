using System;

namespace Vision.MultiStream.Inference.Services.Audio
{
    /// <summary>
    /// 디코더에서 만들어진 PCM 청크를 받아 스피커로 흘려보내는 컴포넌트.
    /// 한 스트림당 하나씩 가지며, 스트림 정지 시 Dispose 한다.
    ///
    /// 구현체는 첫 Push 시 포맷(sample rate / channels) 을 결정하고,
    /// 같은 포맷이 들어오는 동안 재사용한다. 포맷이 바뀌면 내부적으로 재초기화.
    /// </summary>
    public interface IAudioOutput : IDisposable
    {
        void Push(AudioFrame frame);

        // 진단 — UI/로그 폴링용. 구현이 아직 초기화 전이면 0/0 리턴.
        int BufferedMs { get; }
        int BufferLengthMs { get; }
        double FillRatio { get; }
        double DropRatio { get; }
        int TotalDroppedMs { get; }
    }
}
