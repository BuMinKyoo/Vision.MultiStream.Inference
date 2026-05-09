using System;
using NAudio.Wave;

namespace Vision.MultiStream.Inference.Services.Audio
{
    /// <summary>
    /// NAudio 의 WaveOutEvent + BufferedWaveProvider 로 PCM 을 출력하는 구현.
    ///
    /// 설계 메모:
    ///   - WASAPI shared 모드(WasapiOut)는 다중 인스턴스 동시 출력 시 종종 init 충돌 →
    ///     멀티스트림 환경에서는 WaveOutEvent 가 더 안전 (윈도우 기본 믹서 경로).
    ///   - BufferedWaveProvider 가 푸시-기반이라 디코더 스레드에서 그냥 AddSamples 호출 가능.
    ///   - 처음 Push 가 들어올 때 WaveFormat 확정 → 이후 포맷 바뀌면 재생성.
    /// </summary>
    public sealed class WasapiAudioOutput : IAudioOutput
    {
        private readonly object _gate = new();

        private WaveOutEvent? _waveOut;
        private BufferedWaveProvider? _buffer;
        private int _sampleRate;
        private int _channels;
        private bool _disposed;

        public void Push(AudioFrame frame)
        {
            if (_disposed)
            {
                return;
            }

            lock (_gate)
            {
                if (_disposed)
                {
                    return;
                }

                // 포맷이 처음 들어오거나 바뀌면 (드물게 RTSP 가 mid-stream 으로 바꿀 수 있음) 출력기 재구축
                if (_waveOut == null || frame.SampleRate != _sampleRate || frame.Channels != _channels)
                {
                    DisposeOutputNoLock();
                    _sampleRate = frame.SampleRate;
                    _channels = frame.Channels;

                    var format = new WaveFormat(_sampleRate, 16, _channels);
                    _buffer = new BufferedWaveProvider(format)
                    {
                        // 1초치 버퍼면 충분. 더 크면 lag, 더 작으면 jitter 시 끊김.
                        BufferDuration = TimeSpan.FromSeconds(1),
                        DiscardOnBufferOverflow = true
                    };
                    _waveOut = new WaveOutEvent
                    {
                        // 50ms 정도면 체감 지연 ~80ms 수준
                        DesiredLatency = 80,
                        NumberOfBuffers = 3
                    };
                    _waveOut.Init(_buffer);
                    _waveOut.Play();
                }

                _buffer!.AddSamples(frame.Pcm, 0, frame.Pcm.Length);
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                _disposed = true;
                DisposeOutputNoLock();
            }
        }

        private void DisposeOutputNoLock()
        {
            try
            {
                _waveOut?.Stop();
            }
            catch
            {
                // 출력 정지 중 발생하는 예외는 무시 (이미 디바이스가 사라진 경우 등)
            }
            _waveOut?.Dispose();
            _waveOut = null;
            _buffer = null;
        }
    }
}
