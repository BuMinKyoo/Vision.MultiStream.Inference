using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Channels;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Services.Audio;
using Vision.MultiStream.Inference.Services.Rtsp.FFmpeg;
using Vision.MultiStream.Inference.Services.Rtsp.Pipeline;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 파이프라인 파사드. 역할별 단계 클래스를 조립·구동한다.
    ///
    ///   [RtspDemuxer] → VideoPacketQueue → [VideoDecoder] → VideoFrameQueue → [VideoRenderer]
    ///                 → AudioPacketQueue → [AudioDecoder] → AudioFrameQueue → [AudioRenderer]
    ///
    /// 각 단계는 별도 스레드에서 돌고, 단계 사이는 MediaQueue 로만 주고받는다.
    /// 비디오/오디오 동기화는 MediaClock(audio-master / wall-clock) 으로 맞춘다.
    /// 공개 API(이벤트·Reader·토글)는 기존과 동일해 호출부 변경이 필요 없다.
    /// </summary>
    public sealed class RtspFrameSource : IDisposable
    {
        private const int VideoPacketQueueCapacity = 50;
        private const int AudioPacketQueueCapacity = 50;
        private const int VideoFrameQueueCapacity = 10;
        private const int AudioFrameQueueCapacity = 30;

        private readonly string _url;
        private readonly Channel<RtspFrame> _channel;
        private readonly MediaClock _clock = new();
        private readonly VideoRenderSettings _settings = new();
        private readonly object _lifecycle = new();

        private CancellationTokenSource? _cts;
        private Thread? _startupThread;
        private volatile bool _isRunning;

        private IAudioOutput? _audioOutput;

        private RtspDemuxer? _demuxer;
        private VideoDecoder? _videoDecoder;
        private VideoRenderer? _videoRenderer;
        private AudioDecoder? _audioDecoder;
        private AudioRenderer? _audioRenderer;

        private BlockingCollection<IntPtr>? _videoPacketQueue;
        private BlockingCollection<IntPtr>? _videoFrameQueue;
        private BlockingCollection<IntPtr>? _audioPacketQueue;
        private BlockingCollection<AudioFrame>? _audioFrameQueue;

        public RtspFrameSource(string rtspUrl)
        {
            _url = rtspUrl;
            _channel = Channel.CreateBounded<RtspFrame>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true
            });
        }

        public ChannelReader<RtspFrame> Reader => _channel.Reader;
        public bool IsRunning => _isRunning;

        public bool ReaderFramesEnabled
        {
            get => _settings.ReaderFramesEnabled;
            set => _settings.ReaderFramesEnabled = value;
        }

        public bool UseYuvDisplayFrames
        {
            get => _settings.UseYuvDisplayFrames;
            set => _settings.UseYuvDisplayFrames = value;
        }

        public event EventHandler<string>? StatusChanged;
        public event EventHandler<RtspFrame>? FrameCaptured;
        public event EventHandler<RtspYuvFrame>? YuvFrameCaptured;

        /// <summary>
        /// 연결·디코딩 시작. avformat_open_input 이 블로킹이므로 조립은 백그라운드 스레드에서 한다.
        /// audioOutput 가 null 이 아니고 스트림에 오디오가 있으면 오디오도 활성화한다.
        /// </summary>
        public void Start(IAudioOutput? audioOutput)
        {
            lock (_lifecycle)
            {
                if (_isRunning)
                {
                    return;
                }

                FFmpegLibraryLoader.EnsureRegistered();

                _audioOutput = audioOutput;
                _clock.Reset();
                _cts = new CancellationTokenSource();
                _isRunning = true;

                CancellationToken token = _cts.Token;
                _startupThread = new Thread(() => Startup(token))
                {
                    IsBackground = true,
                    Name = $"RtspStartup[{_url}]"
                };
                _startupThread.Start();
            }
        }

        // 백그라운드 조립: 연결 → 스트림 탐색 → 단계 생성/오픈 → 스레드 기동.
        private void Startup(CancellationToken token)
        {
            RtspDemuxer? demuxer = null;
            VideoDecoder? videoDecoder = null;
            VideoRenderer? videoRenderer = null;
            AudioDecoder? audioDecoder = null;
            AudioRenderer? audioRenderer = null;

            try
            {
                demuxer = new RtspDemuxer(_url, RaiseStatus);
                RtspStreamInfo info = demuxer.Open();

                if (token.IsCancellationRequested)
                {
                    demuxer.Dispose();
                    return;
                }

                if (!info.HasVideo)
                {
                    RaiseStatus("열 수 있는 스트림 없음");
                    demuxer.Dispose();
                    _channel.Writer.TryComplete();
                    return;
                }

                // 바운디드 BlockingCollection: 가득 차면 생산자가 대기(backpressure).
                // 멀쩡한 프레임을 큐가 버리지 않으며, 지연 제어는 VideoRenderer 의 MediaClock
                // 게이팅이 담당(늦으면 drop, 이르면 sleep). 렌더러 페이싱이 그대로 디코더→디먹서로
                // backpressure 되어 기존 단일 스레드와 같은 부드러운 출력이 된다.
                var videoPacketQueue = new BlockingCollection<IntPtr>(
                    new ConcurrentQueue<IntPtr>(), VideoPacketQueueCapacity);
                var videoFrameQueue = new BlockingCollection<IntPtr>(
                    new ConcurrentQueue<IntPtr>(), VideoFrameQueueCapacity);

                videoDecoder = new VideoDecoder(info.VideoCodecParams, videoPacketQueue, videoFrameQueue, RaiseStatus);
                if (!videoDecoder.Open())
                {
                    RaiseStatus("비디오 코덱 열기 실패");
                    videoDecoder.Dispose();
                    demuxer.Dispose();
                    _channel.Writer.TryComplete();
                    return;
                }

                videoRenderer = new VideoRenderer(
                    videoFrameQueue, info.VideoTimeBase, _clock, _settings,
                    _channel.Writer, RaiseFrameCaptured, RaiseYuvCaptured, RaiseStatus);

                BlockingCollection<IntPtr>? audioPacketQueue = null;
                BlockingCollection<AudioFrame>? audioFrameQueue = null;
                bool audioActive = info.HasAudio && _audioOutput != null;

                if (audioActive)
                {
                    audioPacketQueue = new BlockingCollection<IntPtr>(
                        new ConcurrentQueue<IntPtr>(), AudioPacketQueueCapacity);
                    audioFrameQueue = new BlockingCollection<AudioFrame>(
                        new ConcurrentQueue<AudioFrame>(), AudioFrameQueueCapacity);

                    audioDecoder = new AudioDecoder(
                        info.AudioCodecParams, info.AudioTimeBase, audioPacketQueue, audioFrameQueue, RaiseStatus);
                    if (audioDecoder.Open())
                    {
                        audioRenderer = new AudioRenderer(audioFrameQueue, _audioOutput!, _clock, RaiseStatus);
                        _clock.UseAudioMaster(_audioOutput!);
                    }
                    else
                    {
                        // 오디오 실패해도 영상은 진행.
                        audioDecoder.Dispose();
                        audioDecoder = null;
                        audioPacketQueue = null;
                        audioFrameQueue = null;
                        audioActive = false;
                    }
                }

                lock (_lifecycle)
                {
                    if (!_isRunning)
                    {
                        // 조립 도중 Stop 호출됨 → 만든 것 정리하고 빠진다.
                        videoRenderer.Dispose();
                        videoDecoder.Dispose();
                        audioDecoder?.Dispose();
                        demuxer.Dispose();
                        return;
                    }

                    _demuxer = demuxer;
                    _videoDecoder = videoDecoder;
                    _videoRenderer = videoRenderer;
                    _audioDecoder = audioDecoder;
                    _audioRenderer = audioRenderer;
                    _videoPacketQueue = videoPacketQueue;
                    _videoFrameQueue = videoFrameQueue;
                    _audioPacketQueue = audioPacketQueue;
                    _audioFrameQueue = audioFrameQueue;
                }

                // 소비자(렌더러) 먼저, 그다음 디코더, 마지막에 생산자(디먹서).
                videoRenderer.Start(token);
                videoDecoder.Start(token);
                audioRenderer?.Start(token);
                audioDecoder?.Start(token);
                demuxer.Start(videoPacketQueue, audioPacketQueue, audioActive, token);

                RaiseStatus(
                    $"연결됨: {_url} ({info.Width}x{info.Height}) +Video" +
                    (audioActive ? " +Audio" : ""));
            }
            catch (Exception ex)
            {
                RaiseStatus($"시작 오류: {ex.Message}");
                videoRenderer?.Dispose();
                videoDecoder?.Dispose();
                audioDecoder?.Dispose();
                demuxer?.Dispose();
                _channel.Writer.TryComplete();
            }
        }

        public void Stop()
        {
            lock (_lifecycle)
            {
                if (!_isRunning)
                {
                    return;
                }
                _isRunning = false;
            }

            _cts?.Cancel();

            // 조립이 끝날(또는 빠질) 때까지 기다린다.
            _startupThread?.Join(TimeSpan.FromSeconds(6));
            _startupThread = null;

            // 셧다운은 생산자 → 소비자 순으로 큐를 닫으며 연쇄적으로 끝낸다.
            _demuxer?.Join(TimeSpan.FromSeconds(3));
            _videoPacketQueue?.CompleteAdding();
            _audioPacketQueue?.CompleteAdding();

            _videoDecoder?.Join(TimeSpan.FromSeconds(2));
            _videoFrameQueue?.CompleteAdding();
            _videoRenderer?.Join(TimeSpan.FromSeconds(2));

            _audioDecoder?.Join(TimeSpan.FromSeconds(2));
            _audioFrameQueue?.CompleteAdding();
            _audioRenderer?.Join(TimeSpan.FromSeconds(2));

            // 큐에 남은 unmanaged 포인터 회수 (오디오 프레임은 managed 라 GC 가 처리).
            FfmpegNative.DrainAndFreePackets(_videoPacketQueue);
            FfmpegNative.DrainAndFreeFrames(_videoFrameQueue);
            FfmpegNative.DrainAndFreePackets(_audioPacketQueue);

            _videoRenderer?.Dispose();
            _videoDecoder?.Dispose();
            _audioDecoder?.Dispose();
            _demuxer?.Dispose();

            _videoRenderer = null;
            _videoDecoder = null;
            _audioDecoder = null;
            _audioRenderer = null;
            _demuxer = null;
            _videoPacketQueue = null;
            _videoFrameQueue = null;
            _audioPacketQueue = null;
            _audioFrameQueue = null;

            _audioOutput?.Dispose();
            _audioOutput = null;

            _channel.Writer.TryComplete();
            RaiseStatus("정지");

            _cts?.Dispose();
            _cts = null;
        }

        // VideoRenderer 가 호출. 구독자가 있으면 프레임을 넘긴 것으로 보고 true 반환
        // (구독자가 unmanaged 버퍼 해제를 책임진다). 구독자 없으면 false → 렌더러가 해제.
        private bool RaiseFrameCaptured(RtspFrame frame)
        {
            EventHandler<RtspFrame>? handler = FrameCaptured;
            if (handler == null)
            {
                return false;
            }
            handler.Invoke(this, frame);
            return true;
        }

        private void RaiseYuvCaptured(RtspYuvFrame frame)
        {
            YuvFrameCaptured?.Invoke(this, frame);
        }

        private void RaiseStatus(string message)
        {
            StatusChanged?.Invoke(this, message);
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
