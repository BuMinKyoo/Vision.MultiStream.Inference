using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using OpenCvSharp;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP URL 에서 프레임을 읽어 1슬롯 latest-only 큐로 노출하는 서비스.
    /// 책임 1개: "수신 → 큐 채우기".
    /// 추론/렌더링 책임은 호출자(ViewModel) 가 가짐.
    ///
    /// 스레드 모델:
    ///   - 자기 전용 백그라운드 스레드 1개에서 VideoCapture.Read() 블로킹 루프
    ///   - Channel 의 FullMode=DropOldest 로 쓰기 비차단 → 소비자가 늦어도 송신 안 막힘
    ///   - 결과: 소비자는 항상 "가장 최신 프레임" 만 보게 됨 (영상 끊김 방지의 핵심)
    /// </summary>
    public sealed class RtspFrameSource : IDisposable
    {
        private readonly string _url;
        private readonly Channel<RtspFrame> _channel;

        private CancellationTokenSource? _cts;
        private Thread? _captureThread;
        private volatile bool _isRunning;

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

        /// <summary>
        /// 소비자(ViewModel) 는 이 Reader 에서 ReadAsync / WaitToReadAsync 로 프레임 받음.
        /// </summary>
        public ChannelReader<RtspFrame> Reader => _channel.Reader;

        public bool IsRunning => _isRunning;

        /// <summary>
        /// 연결/오류/종료 상태 변화 알림 (UI 에 출력하기 좋게).
        /// </summary>
        public event EventHandler<string>? StatusChanged;

        /// <summary>
        /// 프레임이 도착할 때마다 캡처 스레드에서 발생.
        /// 영상 표시(WriteableBitmap)용 — 모든 프레임을 받음. 핸들러에서 무거운 작업 금지.
        /// 추론은 별도로 Reader(채널)에서 latest-only 로 받을 것.
        /// </summary>
        public event EventHandler<RtspFrame>? FrameCaptured;

        public void Start()
        {
            if (_isRunning)
            {
                return;
            }

            _cts = new CancellationTokenSource();

            // VideoCapture.Read()가 블로킹 호출이라 Task(ThreadPool)가 아닌 전용 Thread 사용
            // ThreadPool 스레드를 영구 점유하면 다른 Task들이 스레드를 못 얻어 굶어 죽음
            _captureThread = new Thread(() => CaptureLoop(_cts.Token))
            {
                IsBackground = true, // 메인 스레드 종료 시 이 스레드도 자동 종료
                Name = $"RtspCapture[{_url}]"
            };
            _isRunning = true;
            _captureThread.Start();
        }

        public void Stop()
        {
            if (!_isRunning)
            {
                return;
            }

            _isRunning = false;
            _cts?.Cancel(); // CaptureLoop의 ct.IsCancellationRequested를 true로 만들어 루프 탈출 유도
            _captureThread?.Join(TimeSpan.FromSeconds(2)); // 캡처 스레드가 완전히 끝날 때까지 최대 2초 대기
            _captureThread = null;
            _cts?.Dispose();
            _cts = null;
        }

        private void CaptureLoop(CancellationToken ct)
        {
            VideoCapture? capture = null;
            Mat? mat = null;
            try
            {
                // VideoCapture: 내부적으로 FFmpeg을 사용해 RTSP 패킷 수신 → 디먹싱 → H.264 디코딩
                capture = new VideoCapture(_url);
                if (!capture.IsOpened())
                {
                    RaiseStatus($"열기 실패: {_url}");
                    return;
                }

                RaiseStatus($"연결됨: {_url} ({capture.FrameWidth}x{capture.FrameHeight} @ {capture.Fps:F1} fps)");

                mat = new Mat(); // 프레임 버퍼 (매 Read마다 재사용)
                int consecutiveFailures = 0;

                // 실제 스트림 fps 기준으로 1초치 프레임 수를 계산
                // fps를 못 읽으면(0 이하) 30fps로 fallback
                double streamFps = capture.Fps > 0 ? capture.Fps : 30.0;
                int maxFailures = (int)Math.Ceiling(streamFps);

                while (!ct.IsCancellationRequested)
                {
                    // Read: 다음 프레임이 올 때까지 블로킹 (fps에 따라 간격 다름)
                    // 디코딩 결과는 BGR row-major 포맷으로 mat에 채워짐
                    if (!capture.Read(mat) || mat.Empty())
                    {
                        consecutiveFailures++;
                        if (consecutiveFailures >= maxFailures)
                        {
                            // 1초치 프레임을 연속으로 못 받으면 연결 끊김으로 판단
                            RaiseStatus("프레임 수신 끊김");
                            break;
                        }
                        Thread.Sleep((int)(1000.0 / streamFps)); // fps에 맞는 대기 시간
                        continue;
                    }

                    consecutiveFailures = 0;

                    int width = mat.Width;
                    int height = mat.Height;
                    int channels = mat.Channels(); // 보통 3(BGR), 드물게 4(BGRA)
                    int byteCount = width * height * channels;

                    // mat.Data: 네이티브(C++) 메모리 주소 (IntPtr)
                    // Marshal.Copy: 네이티브 메모리 → 관리 메모리(byte[])로 복사
                    // 복사하지 않으면 다음 Read()에서 mat 버퍼가 덮어써져 데이터가 깨짐
                    var bgr = new byte[byteCount];
                    Marshal.Copy(mat.Data, bgr, 0, byteCount);

                    var frame = new RtspFrame(bgr, width, height, DateTime.UtcNow);

                    // 영상 표시용: 모든 프레임을 이벤트로 즉시 알림 (구독자가 Dispatcher로 마샬링할 책임)
                    FrameCaptured?.Invoke(this, frame);

                    // 추론용: 1슬롯 큐에 넣음. DropOldest 정책이라 이전 프레임은 자동 폐기
                    // → 추론 루프가 느려도 항상 최신 프레임만 처리하게 됨
                    _channel.Writer.TryWrite(frame);
                }
            }
            catch (Exception ex)
            {
                RaiseStatus($"캡처 오류: {ex.Message}");
            }
            finally
            {
                mat?.Dispose();
                capture?.Dispose();
                _channel.Writer.TryComplete(); // 채널을 닫아 추론 루프의 ReadAllAsync가 정상 종료되게 함
                RaiseStatus("정지");
            }
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
