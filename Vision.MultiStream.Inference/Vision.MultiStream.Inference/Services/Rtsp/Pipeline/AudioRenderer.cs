using System;
using System.Collections.Concurrent;
using System.Threading;
using Vision.MultiStream.Inference.Services.Audio;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 파이프라인 3단계(오디오): 디코딩된 PCM 프레임을 사운드카드로 push 하고
    /// MediaClock 을 audio-master 위치로 갱신한다(게이팅 없음 — 사운드카드가 페이싱).
    /// </summary>
    internal sealed class AudioRenderer
    {
        private readonly BlockingCollection<AudioFrame> _frameQueue;
        private readonly IAudioOutput _output;
        private readonly MediaClock _clock;
        private readonly Action<string> _onStatus;

        private Thread? _thread;

        public AudioRenderer(
            BlockingCollection<AudioFrame> frameQueue,
            IAudioOutput output,
            MediaClock clock,
            Action<string> onStatus)
        {
            _frameQueue = frameQueue;
            _output = output;
            _clock = clock;
            _onStatus = onStatus;
        }

        public void Start(CancellationToken ct)
        {
            _thread = new Thread(() => RenderLoop(ct))
            {
                IsBackground = true,
                Name = "RtspAudioRender"
            };
            _thread.Start();
        }

        private void RenderLoop(CancellationToken ct)
        {
            try
            {
                foreach (AudioFrame frame in _frameQueue.GetConsumingEnumerable(ct))
                {
                    _output.Push(frame);
                    if (!double.IsNaN(frame.PtsSeconds))
                    {
                        _clock.OnAudioPushed(frame.PtsSeconds);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex)
            {
                _onStatus($"오디오 렌더러 오류: {ex.Message}");
            }
        }

        public void Join(TimeSpan timeout)
        {
            _thread?.Join(timeout);
            _thread = null;
        }
    }
}
