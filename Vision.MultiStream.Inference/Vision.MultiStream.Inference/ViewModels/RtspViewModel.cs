using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vision.MultiStream.Inference.Services.Yolo;

namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// RTSP 스트림 객체 검출 ViewModel.
    /// 책임: RtspFrameSource(수신) + IRtspFrameDetector(추론) 의 결과를 UI 바인딩으로 노출.
    /// 스레드 분리:
    ///   - 영상 표시: RtspFrameSource.FrameCaptured (캡처 스레드) → Dispatcher.BeginInvoke → WriteableBitmap
    ///   - 추론     : RtspFrameSource.Reader (latest-only) → Task → Dispatcher.BeginInvoke → Detections
    /// </summary>
    public class RtspViewModel : BaseViewModel, IDisposable
    {
        private readonly IRtspFrameDetector _cpuDetector;
        private readonly IRtspFrameDetector _dmlDetector;
        private readonly IRtspFrameDetector _gpuDetector;
        private readonly Dispatcher _dispatcher;

        private string _rtspUrl = "rtsp://localhost:8554/cam1";
        private bool _isStreaming;
        private InferenceDevice _selectedDevice = InferenceDevice.Cpu;
        private string _statusMessage = "RTSP URL 입력 후 [연결] 버튼을 누르세요.";
        private WriteableBitmap? _imageSource;
        private int _imageWidth;
        private int _imageHeight;
        private double _displayFps;
        private double _inferenceFps;
        private double _preprocessMs;
        private double _inferenceMs;
        private double _postprocessMs;

        private RtspFrameSource? _source;
        private CancellationTokenSource? _cts;
        private Task? _inferenceTask;

        private readonly FpsCounter _displayFpsCounter = new();
        private readonly FpsCounter _inferenceFpsCounter = new();

        public RtspViewModel(IRtspFrameDetector cpuDetector, IRtspFrameDetector dmlDetector, IRtspFrameDetector gpuDetector)
        {
            _cpuDetector = cpuDetector;
            _dmlDetector = dmlDetector;
            _gpuDetector = gpuDetector;
            _dispatcher = Application.Current.Dispatcher;
            ConnectCommand = new RelayCommand(Connect, () => !IsStreaming && !string.IsNullOrWhiteSpace(RtspUrl));
            DisconnectCommand = new RelayCommand(Disconnect, () => IsStreaming);
        }

        // 연결 시 선택된 디바이스에 해당하는 디텍터 반환
        private IRtspFrameDetector ActiveDetector => _selectedDevice switch
        {
            InferenceDevice.DirectML => _dmlDetector,
            InferenceDevice.Gpu     => _gpuDetector,
            _                       => _cpuDetector,
        };

        // RadioButton 바인딩용 프로퍼티 (WPF에서 enum을 RadioButton에 직접 바인딩하기 번거로워 bool 3개로 노출)
        public bool UseCpu
        {
            get => _selectedDevice == InferenceDevice.Cpu;
            set { if (value) { _selectedDevice = InferenceDevice.Cpu; NotifyDeviceChanged(); } }
        }

        public bool UseDirectML
        {
            get => _selectedDevice == InferenceDevice.DirectML;
            set { if (value) { _selectedDevice = InferenceDevice.DirectML; NotifyDeviceChanged(); } }
        }

        public bool UseGpu
        {
            get => _selectedDevice == InferenceDevice.Gpu;
            set { if (value) { _selectedDevice = InferenceDevice.Gpu; NotifyDeviceChanged(); } }
        }

        private void NotifyDeviceChanged()
        {
            OnPropertyChanged(nameof(UseCpu));
            OnPropertyChanged(nameof(UseDirectML));
            OnPropertyChanged(nameof(UseGpu));
        }

        public ObservableCollection<Detection> Detections { get; } = new();

        public RelayCommand ConnectCommand { get; }
        public RelayCommand DisconnectCommand { get; }

        public string RtspUrl
        {
            get => _rtspUrl;
            set
            {
                if (_rtspUrl == value)
                {
                    return;
                }
                _rtspUrl = value;
                OnPropertyChanged();
                ConnectCommand.RaiseCanExecuteChanged();
            }
        }

        public bool IsStreaming
        {
            get => _isStreaming;
            private set
            {
                if (_isStreaming == value)
                {
                    return;
                }
                _isStreaming = value;
                OnPropertyChanged();
                ConnectCommand.RaiseCanExecuteChanged();
                DisconnectCommand.RaiseCanExecuteChanged();
            }
        }

        public string StatusMessage
        {
            get => _statusMessage;
            private set
            {
                if (_statusMessage == value)
                {
                    return;
                }
                _statusMessage = value;
                OnPropertyChanged();
            }
        }

        public WriteableBitmap? ImageSource
        {
            get => _imageSource;
            private set
            {
                if (_imageSource == value)
                {
                    return;
                }
                _imageSource = value;
                OnPropertyChanged();
            }
        }

        public int ImageWidth
        {
            get => _imageWidth;
            private set
            {
                if (_imageWidth == value)
                {
                    return;
                }
                _imageWidth = value;
                OnPropertyChanged();
            }
        }

        public int ImageHeight
        {
            get => _imageHeight;
            private set
            {
                if (_imageHeight == value)
                {
                    return;
                }
                _imageHeight = value;
                OnPropertyChanged();
            }
        }

        public double DisplayFps
        {
            get => _displayFps;
            private set
            {
                if (Math.Abs(_displayFps - value) < 0.01)
                {
                    return;
                }
                _displayFps = value;
                OnPropertyChanged();
            }
        }

        public double InferenceFps
        {
            get => _inferenceFps;
            private set
            {
                if (Math.Abs(_inferenceFps - value) < 0.01)
                {
                    return;
                }
                _inferenceFps = value;
                OnPropertyChanged();
            }
        }

        public double PreprocessMs
        {
            get => _preprocessMs;
            private set
            {
                if (Math.Abs(_preprocessMs - value) < 0.05)
                {
                    return;
                }
                _preprocessMs = value;
                OnPropertyChanged();
            }
        }

        public double InferenceMs
        {
            get => _inferenceMs;
            private set
            {
                if (Math.Abs(_inferenceMs - value) < 0.05)
                {
                    return;
                }
                _inferenceMs = value;
                OnPropertyChanged();
            }
        }

        public double PostprocessMs
        {
            get => _postprocessMs;
            private set
            {
                if (Math.Abs(_postprocessMs - value) < 0.05)
                {
                    return;
                }
                _postprocessMs = value;
                OnPropertyChanged();
            }
        }

        private void Connect()
        {
            if (IsStreaming)
            {
                return;
            }

            try
            {
                // ViewModel 생성 시점부터 타이머가 흐르고 있으므로 연결 시점에 리셋
                _displayFpsCounter.Reset();
                _inferenceFpsCounter.Reset();

                // RtspFrameSource: RTSP 수신 + 채널 관리 담당 객체
                _source = new RtspFrameSource(RtspUrl);

                // 이벤트 구독: 발생 시 호출할 메서드를 등록 (메서드 그룹 방식)
                // StatusChanged  → RTSP 연결 상태 문자열을 UI에 표시
                // FrameCaptured  → 디코딩된 프레임을 WriteableBitmap에 렌더링
                _source.StatusChanged += OnSourceStatusChanged;
                _source.FrameCaptured += OnFrameCapturedForDisplay;

                // 전용 백그라운드 스레드 시작 (VideoCapture.Read가 블로킹이라 ThreadPool 대신 전용 Thread 사용)
                _source.Start();

                // 추론 루프는 ThreadPool Task로 실행 (채널에서 프레임 꺼내 → YOLO 추론 → UI 갱신 반복)
                _cts = new CancellationTokenSource();
                _inferenceTask = Task.Run(() => InferenceLoopAsync(_cts.Token));

                IsStreaming = true;
                StatusMessage = "연결 중...";
            }
            catch (Exception ex)
            {
                StatusMessage = $"연결 실패: {ex.Message}";
                Disconnect();
            }
        }

        private void Disconnect()
        {
            if (!IsStreaming && _source == null)
            {
                return;
            }

            try
            {
                // 추론 루프에 취소 신호 전달 → InferenceLoopAsync의 ct.IsCancellationRequested가 true가 됨
                _cts?.Cancel();

                if (_source != null)
                {
                    // 이벤트 구독 해제: 메서드 그룹으로 등록했기 때문에 동일한 메서드로 정확히 제거 가능
                    // 해제하지 않으면 _source가 살아있는 한 핸들러가 계속 호출됨 (메모리 누수)
                    _source.FrameCaptured -= OnFrameCapturedForDisplay;
                    _source.StatusChanged -= OnSourceStatusChanged;
                    _source.Stop();
                    _source.Dispose();
                    _source = null;
                }

                _inferenceTask = null;
                _cts?.Dispose();
                _cts = null;

                IsStreaming = false;
                Detections.Clear();
                DisplayFps = 0;
                InferenceFps = 0;
                PreprocessMs = 0;
                InferenceMs = 0;
                PostprocessMs = 0;
            }
            catch (Exception ex)
            {
                StatusMessage = $"중지 중 오류: {ex.Message}";
            }
        }

        // RtspFrameSource가 상태 변경 시 캡처 스레드에서 호출 → UI 스레드로 마샬링
        private void OnSourceStatusChanged(object? sender, string message)
        {
            _dispatcher.BeginInvoke(() => StatusMessage = message);
        }

        // 캡처 스레드에서 프레임마다 호출됨
        private void OnFrameCapturedForDisplay(object? sender, RtspFrame frame)
        {
            // FPS 측정은 캡처 스레드에서: UI 스레드 대기 시간이 섞이면 실제 수신 FPS가 아닌 렌더링 FPS가 됨
            _displayFpsCounter.Tick(out double fps);

            // WriteableBitmap은 UI 스레드에서만 접근 가능 → Dispatcher로 마샬링
            // BeginInvoke(비동기): 캡처 스레드가 렌더링을 기다리지 않고 즉시 다음 프레임으로 진행 → 영상 끊김 방지
            // DispatcherPriority.Render: 일반 Background 작업보다 높은 우선순위로 처리
            _dispatcher.BeginInvoke(() =>
            {
                RenderFrameToBitmap(frame);
                DisplayFps = fps;
            }, DispatcherPriority.Render);
        }

        // UI 스레드에서만 호출됨
        private void RenderFrameToBitmap(RtspFrame frame)
        {
            // 첫 프레임이거나 해상도가 바뀌면 WriteableBitmap 새로 생성
            if (_imageSource == null
                || _imageSource.PixelWidth != frame.Width
                || _imageSource.PixelHeight != frame.Height)
            {
                // Bgr24: OpenCV에서 온 BGR 픽셀을 그대로 쓸 수 있어 RGB 스왑 불필요
                ImageSource = new WriteableBitmap(
                    frame.Width, frame.Height, 96, 96, PixelFormats.Bgr24, null);
                ImageWidth = frame.Width;
                ImageHeight = frame.Height;
            }

            // stride: 한 행의 바이트 수 (픽셀 너비 × 채널 수)
            int stride = frame.Width * 3;
            _imageSource!.WritePixels(
                new Int32Rect(0, 0, frame.Width, frame.Height),
                frame.BgrPixels,
                stride,
                0);
        }

        // ThreadPool Task에서 실행되는 추론 루프
        private async Task InferenceLoopAsync(CancellationToken ct)
        {
            if (_source == null)
            {
                return;
            }

            try
            {
                // ReadAllAsync: 1슬롯 채널에서 프레임이 올 때까지 비동기 대기 후 꺼냄
                // DropOldest 정책이라 추론이 느려도 항상 가장 최신 프레임만 처리
                await foreach (RtspFrame frame in _source.Reader.ReadAllAsync(ct))
                {
                    if (ct.IsCancellationRequested)
                    {
                        break;
                    }

                    var (detections, timings) = await ActiveDetector
                        .DetectAsync(frame.BgrPixels, frame.Width, frame.Height, ct)
                        .ConfigureAwait(false);

                    _inferenceFpsCounter.Tick(out double fps);

                    _ = _dispatcher.BeginInvoke(() =>
                    {
                        Detections.Clear();
                        foreach (Detection d in detections)
                        {
                            Detections.Add(d);
                        }
                        InferenceFps = fps;
                        PreprocessMs = timings.PreprocessMs;
                        InferenceMs = timings.InferenceMs;
                        PostprocessMs = timings.PostprocessMs;
                    }, DispatcherPriority.Background);
                }
            }
            catch (OperationCanceledException)
            {
                // Disconnect()에서 _cts.Cancel() 호출 시 정상적으로 발생하는 예외
            }
            catch (Exception ex)
            {
                _ = _dispatcher.BeginInvoke(() => StatusMessage = $"추론 오류: {ex.Message}");
            }
        }

        public void Dispose()
        {
            Disconnect();
        }

        // 1초 슬라이딩 윈도우 FPS 카운터
        private sealed class FpsCounter
        {
            private readonly Stopwatch _sw = Stopwatch.StartNew(); // 생성 즉시 시작
            private int _count;        // 마지막 계산 이후 Tick 호출 횟수
            private long _lastReportMs; // 마지막으로 FPS를 계산했던 시각 (ms)
            private double _lastFps;   // 마지막으로 계산된 FPS 값

            // 연결 시점에 호출해 타이머와 카운터를 초기화
            public void Reset()
            {
                _sw.Restart();
                _count = 0;
                _lastReportMs = 0;
                _lastFps = 0;
            }

            public void Tick(out double fps)
            {
                _count++;
                long elapsed = _sw.ElapsedMilliseconds - _lastReportMs;

                // 1초가 지났을 때만 FPS 재계산 (그 사이에는 직전 값을 그대로 반환)
                if (elapsed >= 1000)
                {
                    _lastFps = _count * 1000.0 / elapsed; // count / (elapsed ms / 1000) = 초당 횟수
                    _count = 0;
                    _lastReportMs = _sw.ElapsedMilliseconds;
                }
                fps = _lastFps;
            }
        }
    }
}
