using System;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

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
    ///
    /// 진단:
    ///   - Push 직전에 BufferedBytes 와 BufferLength 를 비교해 들어가지 못할 양(willDrop) 을 누적.
    ///   - WASAPI 가 그 짧은 시간 동안 일부 빼갈 수 있어 실제 drop 의 상한(overestimate) 임.
    /// </summary>
    public sealed class WasapiAudioOutput : IAudioOutput
    {
        private readonly object _gate = new();

        private WaveOutEvent? _waveOut;
        private BufferedWaveProvider? _buffer;
        private VolumeWaveProvider16? _volumeProvider;
        private int _sampleRate;
        private int _channels;
        private bool _disposed;
        private bool _isMuted;

        // 진단용 누적 카운터.
        private long _totalRequestedBytes;
        private long _totalDroppedBytes;

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
                        BufferDuration = TimeSpan.FromMilliseconds(2000),
                        DiscardOnBufferOverflow = true
                    };
                    // waveOutSetVolume은 디바이스 전체에 적용되어 다른 스트림까지 영향 →
                    // 스트림별 독립 볼륨을 위해 소프트웨어 볼륨 체인 사용
                    _volumeProvider = new VolumeWaveProvider16(_buffer)
                    {
                        Volume = _isMuted ? 0f : 1f
                    };
                    _waveOut = new WaveOutEvent
                    {
                        // 50ms 정도면 체감 지연 ~80ms 수준
                        DesiredLatency = 80,
                        NumberOfBuffers = 3
                    };
                    _waveOut.Init(_volumeProvider);
                    _waveOut.Play();
                }

                int available = _buffer!.BufferLength - _buffer.BufferedBytes;
                int willDrop = Math.Max(0, frame.Pcm.Length - available);
                _totalRequestedBytes += frame.Pcm.Length;
                _totalDroppedBytes += willDrop;

                _buffer.AddSamples(frame.Pcm, 0, frame.Pcm.Length);
            }
        }

        public bool IsMuted
        {
            get
            {
                lock (_gate)
                {
                    return _isMuted;
                }
            }
            set
            {
                lock (_gate)
                {
                    if (_isMuted == value)
                    {
                        return;
                    }
                    _isMuted = value;
                    if (_volumeProvider != null)
                    {
                        _volumeProvider.Volume = value ? 0f : 1f;
                    }
                }
            }
        }

        public int BufferedMs
        {
            get
            {
                lock (_gate)
                {
                    if (_buffer == null)
                    {
                        return 0;
                    }
                    return (int)_buffer.BufferedDuration.TotalMilliseconds;
                }
            }
        }

        public int BufferLengthMs
        {
            get
            {
                lock (_gate)
                {
                    if (_buffer == null)
                    {
                        return 0;
                    }
                    return (int)_buffer.BufferDuration.TotalMilliseconds;
                }
            }
        }

        public double FillRatio
        {
            get
            {
                lock (_gate)
                {
                    if (_buffer == null || _buffer.BufferLength == 0)
                    {
                        return 0;
                    }
                    return (double)_buffer.BufferedBytes / _buffer.BufferLength;
                }
            }
        }

        public long TotalRequestedBytes => System.Threading.Interlocked.Read(ref _totalRequestedBytes);

        public long TotalDroppedBytes => System.Threading.Interlocked.Read(ref _totalDroppedBytes);

        public double DropRatio
        {
            get
            {
                long req = TotalRequestedBytes;
                if (req == 0)
                {
                    return 0;
                }
                return (double)TotalDroppedBytes / req;
            }
        }

        // bytes → ms 환산. WaveFormat 이 결정되기 전엔 0.
        public int TotalDroppedMs
        {
            get
            {
                lock (_gate)
                {
                    if (_buffer == null || _sampleRate == 0 || _channels == 0)
                    {
                        return 0;
                    }
                    int bytesPerSec = _sampleRate * _channels * 2; // S16 = 2 bytes/sample
                    if (bytesPerSec == 0)
                    {
                        return 0;
                    }
                    return (int)(_totalDroppedBytes * 1000 / bytesPerSec);
                }
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
            _volumeProvider = null;
            _buffer = null;
        }
    }
}
