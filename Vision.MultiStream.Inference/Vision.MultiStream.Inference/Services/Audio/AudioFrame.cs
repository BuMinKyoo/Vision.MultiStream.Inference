using System;

namespace Vision.MultiStream.Inference.Services.Audio
{
    /// <summary>
    /// 오디오 디코더 스레드가 만들어 출력기로 넘기는 PCM 청크.
    /// 항상 16-bit signed little-endian 인터리브드 PCM 으로 정규화돼서 오는 것을 가정한다
    /// (= NAudio WaveFormat.CreateIeeeFloatWaveFormat 대신 PCM s16 으로 통일).
    /// 채널 수와 샘플레이트는 스트림마다 다를 수 있으므로 frame 자체에 포함.
    /// </summary>
    public sealed class AudioFrame
    {
        public AudioFrame(byte[] pcm, int sampleRate, int channels, double ptsSeconds)
        {
            Pcm = pcm;
            SampleRate = sampleRate;
            Channels = channels;
            PtsSeconds = ptsSeconds;
        }

        public byte[] Pcm { get; }
        public int SampleRate { get; }
        public int Channels { get; }

        // PTS 가 NOPTS 또는 미지원이면 double.NaN.
        public double PtsSeconds { get; }
    }
}
