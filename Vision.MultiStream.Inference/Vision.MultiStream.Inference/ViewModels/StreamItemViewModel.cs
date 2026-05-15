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
using Vision.MultiStream.Inference.Services.Audio;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vision.MultiStream.Inference.Services.Yolo;

namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 멀티스트림에서 1개 RTSP 스트림을 표현하는 ViewModel.
    /// 책임: 자기 자신의 RtspFrameSource + 추론 루프 + 오디오 출력 수명 관리.
    ///
    /// 세 개의 토글:
    ///   - IsVideoEnabled     : 비디오 디코딩 + 표시. 토글 시 RTSP 재구성.
    ///   - IsAudioEnabled     : 사운드카드 mute/unmute (비디오가 켜져 있을 때만 켤 수 있음 — 비디오에 종속, audio-only 모드 없음).
    ///                          오디오 디코더는 비디오 가동 중에는 항상 돌아가고 PTS 도 계속 push 됨 (audio-master 동기화 유지).
    ///                          토글은 사운드카드 볼륨만 0/1 로 바꿔 RTSP 재구성을 일으키지 않는다.
    ///   - IsInferenceEnabled : YOLO 추론 루프 ON/OFF (비디오가 켜져 있을 때만 의미 있음).
    /// 비디오를 끄면 오디오도 함께 꺼진다.
    /// 추론 토글은 RTSP 를 건드리지 않고 추론 루프만 start/stop 한다.
    /// </summary>
    public sealed class StreamItemViewModel : BaseViewModel, IDisposable
    {
        private readonly Func<InferenceDevice, IRtspFrameDetector> _detectorResolver;
        private readonly Action<StreamItemViewModel> _onRemoveRequested;
        private readonly Dispatcher _dispatcher;

        private string _name;
        private string _rtspUrl;
        private InferenceDevice _device;
        private bool _isVideoEnabled;
        private bool _isAudioEnabled = true;
        private bool _isInferenceEnabled;
        private string _statusMessage = "대기";
        private WriteableBitmap? _imageSource;
        private int _imageWidth;
        private int _imageHeight;
        private double _displayFps;
        private double _inferenceFps;
        private double _preprocessMs;
        private double _inferenceMs;
        private double _postprocessMs;
        private int _audioBufferedMs;
        private int _audioBufferLengthMs;
        private double _audioFillRatio;
        private double _audioDropPercent;
        private int _audioTotalDroppedMs;

        private RtspFrameSource? _source;
        private IAudioOutput? _audioOutput;
        private CancellationTokenSource? _inferenceCts;
        private Task? _inferenceTask;
        private DispatcherTimer? _audioDiagTimer;

        private readonly FpsCounter _displayFpsCounter = new();
        private readonly FpsCounter _inferenceFpsCounter = new();

        public StreamItemViewModel(
            string name,
            string rtspUrl,
            InferenceDevice device,
            Func<InferenceDevice, IRtspFrameDetector> detectorResolver,
            Action<StreamItemViewModel> onRemoveRequested,
            bool initialInferenceEnabled = true)
        {
            _name = name;
            _rtspUrl = rtspUrl;
            _device = device;
            _isInferenceEnabled = initialInferenceEnabled;
            _detectorResolver = detectorResolver;
            _onRemoveRequested = onRemoveRequested;
            _dispatcher = Application.Current.Dispatcher;

            ToggleVideoCommand = new RelayCommand(ToggleVideo, () => !string.IsNullOrWhiteSpace(RtspUrl));
            ToggleAudioCommand = new RelayCommand(ToggleAudio, () => _isVideoEnabled);
            ToggleInferenceCommand = new RelayCommand(ToggleInference);
            StartCommand = new RelayCommand(StartAll, () => !string.IsNullOrWhiteSpace(RtspUrl));
            StopCommand = new RelayCommand(StopAll, () => IsActive);
            RemoveCommand = new RelayCommand(() =>
            {
                StopAll();
                _onRemoveRequested(this);
            });
        }

        public ObservableCollection<Detection> Detections { get; } = new();

        public RelayCommand ToggleVideoCommand { get; }
        public RelayCommand ToggleAudioCommand { get; }
        public RelayCommand ToggleInferenceCommand { get; }
        public RelayCommand StartCommand { get; }
        public RelayCommand StopCommand { get; }
        public RelayCommand RemoveCommand { get; }

        public string Name
        {
            get => _name;
            set
            {
                if (_name != value)
                {
                    _name = value;
                    OnPropertyChanged();
                }
            }
        }

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
                ToggleVideoCommand.RaiseCanExecuteChanged();
            }
        }

        public InferenceDevice Device
        {
            get => _device;
            set
            {
                if (_device == value)
                {
                    return;
                }
                _device = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(UseCpu));
                OnPropertyChanged(nameof(UseDirectML));
                OnPropertyChanged(nameof(UseGpu));
                OnPropertyChanged(nameof(DeviceLabel));
            }
        }

        public string DeviceLabel => _device switch
        {
            InferenceDevice.DirectML => "DML",
            InferenceDevice.Gpu => "CUDA",
            _ => "CPU"
        };

        public bool UseCpu
        {
            get => _device == InferenceDevice.Cpu;
            set
            {
                if (value)
                {
                    Device = InferenceDevice.Cpu;
                }
            }
        }

        public bool UseDirectML
        {
            get => _device == InferenceDevice.DirectML;
            set
            {
                if (value)
                {
                    Device = InferenceDevice.DirectML;
                }
            }
        }

        public bool UseGpu
        {
            get => _device == InferenceDevice.Gpu;
            set
            {
                if (value)
                {
                    Device = InferenceDevice.Gpu;
                }
            }
        }

        public bool IsVideoEnabled
        {
            get => _isVideoEnabled;
            private set
            {
                if (_isVideoEnabled == value)
                {
                    return;
                }
                _isVideoEnabled = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(IsActive));
                StopCommand.RaiseCanExecuteChanged();
            }
        }

        public bool IsAudioEnabled
        {
            get => _isAudioEnabled;
            private set
            {
                if (_isAudioEnabled == value)
                {
                    return;
                }
                _isAudioEnabled = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(IsActive));
                StopCommand.RaiseCanExecuteChanged();
            }
        }

        public bool IsInferenceEnabled
        {
            get => _isInferenceEnabled;
            private set
            {
                if (_isInferenceEnabled == value)
                {
                    return;
                }
                _isInferenceEnabled = value;
                OnPropertyChanged();
            }
        }

        public bool IsActive => _isVideoEnabled || _isAudioEnabled;

        public string StatusMessage
        {
            get => _statusMessage;
            private set
            {
                if (_statusMessage != value)
                {
                    _statusMessage = value;
                    OnPropertyChanged();
                }
            }
        }

        public WriteableBitmap? ImageSource
        {
            get => _imageSource;
            private set
            {
                if (_imageSource != value)
                {
                    _imageSource = value;
                    OnPropertyChanged();
                }
            }
        }

        public int ImageWidth
        {
            get => _imageWidth;
            private set
            {
                if (_imageWidth != value)
                {
                    _imageWidth = value;
                    OnPropertyChanged();
                }
            }
        }

        public int ImageHeight
        {
            get => _imageHeight;
            private set
            {
                if (_imageHeight != value)
                {
                    _imageHeight = value;
                    OnPropertyChanged();
                }
            }
        }

        public double DisplayFps
        {
            get => _displayFps;
            private set
            {
                if (Math.Abs(_displayFps - value) > 0.01)
                {
                    _displayFps = value;
                    OnPropertyChanged();
                }
            }
        }

        public double InferenceFps
        {
            get => _inferenceFps;
            private set
            {
                if (Math.Abs(_inferenceFps - value) > 0.01)
                {
                    _inferenceFps = value;
                    OnPropertyChanged();
                }
            }
        }

        public double PreprocessMs
        {
            get => _preprocessMs;
            private set
            {
                if (Math.Abs(_preprocessMs - value) > 0.05)
                {
                    _preprocessMs = value;
                    OnPropertyChanged();
                }
            }
        }

        public double InferenceMs
        {
            get => _inferenceMs;
            private set
            {
                if (Math.Abs(_inferenceMs - value) > 0.05)
                {
                    _inferenceMs = value;
                    OnPropertyChanged();
                }
            }
        }

        public double PostprocessMs
        {
            get => _postprocessMs;
            private set
            {
                if (Math.Abs(_postprocessMs - value) > 0.05)
                {
                    _postprocessMs = value;
                    OnPropertyChanged();
                }
            }
        }

        public int AudioBufferedMs
        {
            get => _audioBufferedMs;
            private set
            {
                if (_audioBufferedMs != value)
                {
                    _audioBufferedMs = value;
                    OnPropertyChanged();
                }
            }
        }

        public int AudioBufferLengthMs
        {
            get => _audioBufferLengthMs;
            private set
            {
                if (_audioBufferLengthMs != value)
                {
                    _audioBufferLengthMs = value;
                    OnPropertyChanged();
                }
            }
        }

        public double AudioFillRatio
        {
            get => _audioFillRatio;
            private set
            {
                if (Math.Abs(_audioFillRatio - value) > 0.005)
                {
                    _audioFillRatio = value;
                    OnPropertyChanged();
                }
            }
        }

        public double AudioDropPercent
        {
            get => _audioDropPercent;
            private set
            {
                if (Math.Abs(_audioDropPercent - value) > 0.01)
                {
                    _audioDropPercent = value;
                    OnPropertyChanged();
                }
            }
        }

        public int AudioTotalDroppedMs
        {
            get => _audioTotalDroppedMs;
            private set
            {
                if (_audioTotalDroppedMs != value)
                {
                    _audioTotalDroppedMs = value;
                    OnPropertyChanged();
                }
            }
        }

        // 외부(MultiStreamViewModel 의 전체 ON/OFF) 에서 사용할 수 있는 직접 setter
        public void SetVideo(bool enabled)
        {
            if (_isVideoEnabled == enabled)
            {
                return;
            }
            _isVideoEnabled = enabled;
            // 오디오는 비디오에 종속 — 비디오를 끄면 오디오도 함께 꺼진다.
            if (!enabled)
            {
                _isAudioEnabled = false;
            }
            OnPropertyChanged(nameof(IsVideoEnabled));
            OnPropertyChanged(nameof(IsAudioEnabled));
            OnPropertyChanged(nameof(IsActive));
            StopCommand.RaiseCanExecuteChanged();
            ToggleAudioCommand.RaiseCanExecuteChanged();
            ApplyState();
        }

        public void SetAudio(bool enabled)
        {
            // 오디오는 비디오에 종속 — 비디오가 꺼져 있으면 오디오만 켤 수 없다.
            if (enabled && !_isVideoEnabled)
            {
                return;
            }
            if (_isAudioEnabled == enabled)
            {
                return;
            }
            _isAudioEnabled = enabled;
            OnPropertyChanged(nameof(IsAudioEnabled));
            OnPropertyChanged(nameof(IsActive));
            StopCommand.RaiseCanExecuteChanged();

            // 사운드카드 볼륨만 토글. RTSP/디코더는 그대로 → 영상 끊김 없음, audio-master 동기화 유지.
            if (_audioOutput != null)
            {
                _audioOutput.IsMuted = !enabled;
            }
        }

        // 추론 토글은 RTSP 재구성 없이 추론 루프만 start/stop 한다.
        public void SetInference(bool enabled)
        {
            if (_isInferenceEnabled == enabled)
            {
                return;
            }
            _isInferenceEnabled = enabled;
            OnPropertyChanged(nameof(IsInferenceEnabled));

            if (_source == null || !_isVideoEnabled)
            {
                return;
            }

            if (enabled)
            {
                StartInferenceLoop();
            }
            else
            {
                StopInferenceLoop();
                _dispatcher.BeginInvoke(() =>
                {
                    Detections.Clear();
                    InferenceFps = 0;
                    PreprocessMs = 0;
                    InferenceMs = 0;
                    PostprocessMs = 0;
                });
            }
        }

        private void ToggleVideo()
        {
            SetVideo(!_isVideoEnabled);
        }

        private void ToggleAudio()
        {
            SetAudio(!_isAudioEnabled);
        }

        private void ToggleInference()
        {
            SetInference(!_isInferenceEnabled);
        }

        private void StartAll()
        {
            bool changed = !_isVideoEnabled || !_isAudioEnabled;
            _isVideoEnabled = true;
            _isAudioEnabled = true;
            if (changed)
            {
                OnPropertyChanged(nameof(IsVideoEnabled));
                OnPropertyChanged(nameof(IsAudioEnabled));
                OnPropertyChanged(nameof(IsActive));
                StopCommand.RaiseCanExecuteChanged();
                ToggleAudioCommand.RaiseCanExecuteChanged();
                ApplyState();
            }
        }

        private void StopAll()
        {
            bool changed = _isVideoEnabled || _isAudioEnabled;
            _isVideoEnabled = false;
            _isAudioEnabled = false;
            if (changed)
            {
                OnPropertyChanged(nameof(IsVideoEnabled));
                OnPropertyChanged(nameof(IsAudioEnabled));
                OnPropertyChanged(nameof(IsActive));
                StopCommand.RaiseCanExecuteChanged();
                ToggleAudioCommand.RaiseCanExecuteChanged();
                ApplyState();
            }
        }

        /// <summary>
        /// IsVideoEnabled 상태에 맞춰 source 를 재구성한다.
        /// IsVideoEnabled 토글 시 Stop → Start 로 RTSP 재구성. IsAudioEnabled 토글은 이 메서드를 거치지 않고 볼륨만 변경한다.
        /// </summary>
        private void ApplyState()
        {
            // 항상 기존 source 닫고 새로 구성
            TearDownSource();

            if (!_isVideoEnabled)
            {
                Detections.Clear();
                ImageSource = null;
                ImageWidth = 0;
                ImageHeight = 0;
                DisplayFps = 0;
                InferenceFps = 0;
                PreprocessMs = 0;
                InferenceMs = 0;
                PostprocessMs = 0;
                StatusMessage = "정지";
                return;
            }

            if (string.IsNullOrWhiteSpace(RtspUrl))
            {
                StatusMessage = "URL 없음";
                _isVideoEnabled = false;
                _isAudioEnabled = false;
                OnPropertyChanged(nameof(IsVideoEnabled));
                OnPropertyChanged(nameof(IsAudioEnabled));
                OnPropertyChanged(nameof(IsActive));
                ToggleAudioCommand.RaiseCanExecuteChanged();
                return;
            }

            try
            {
                _displayFpsCounter.Reset();
                _inferenceFpsCounter.Reset();

                _source = new RtspFrameSource(RtspUrl);
                _source.StatusChanged += OnSourceStatusChanged;
                _source.FrameCaptured += OnFrameCapturedForDisplay;

                // 오디오 출력은 항상 생성. 토글 OFF 상태면 muted 로 시작.
                // 디코더는 스트림에 오디오 트랙이 있으면 가동, 없으면 자동으로 안 띄움.
                IAudioOutput audioOutput = new WasapiAudioOutput
                {
                    IsMuted = !_isAudioEnabled
                };
                _audioOutput = audioOutput;
                _source.Start(audioOutput);

                if (_isInferenceEnabled)
                {
                    StartInferenceLoop();
                }

                // 오디오 진단 정보를 UI에 주기적으로 갱신하는 타이머
                StartAudioDiagTimer();

                StatusMessage = "연결 중...";
            }
            catch (Exception ex)
            {
                StatusMessage = $"연결 실패: {ex.Message}";
                TearDownSource();
            }
        }

        // RTSP 는 그대로 두고 추론 루프만 가동한다. ApplyState 와 SetInference(true) 양쪽에서 사용.
        private void StartInferenceLoop()
        {
            if (_inferenceTask != null)
            {
                return;
            }
            _inferenceCts = new CancellationTokenSource();
            CancellationToken token = _inferenceCts.Token;
            _inferenceTask = Task.Run(() => InferenceLoopAsync(token));
        }

        // 추론 루프만 정지. RTSP / 디코더는 건드리지 않는다.
        private void StopInferenceLoop()
        {
            try
            {
                _inferenceCts?.Cancel();
            }
            catch
            {
            }
            _inferenceTask = null;
            _inferenceCts?.Dispose();
            _inferenceCts = null;
        }

        private void TearDownSource()
        {
            try
            {
                StopInferenceLoop();
                StopAudioDiagTimer();
                _audioOutput = null; // RtspFrameSource.Stop() 이 Dispose 까지 책임짐

                if (_source != null)
                {
                    _source.FrameCaptured -= OnFrameCapturedForDisplay;
                    _source.StatusChanged -= OnSourceStatusChanged;
                    _source.Stop();
                    _source.Dispose();
                    _source = null;
                }

                AudioBufferedMs = 0;
                AudioBufferLengthMs = 0;
                AudioFillRatio = 0;
                AudioDropPercent = 0;
                AudioTotalDroppedMs = 0;
            }
            catch
            {
                // teardown 중 예외는 무시
            }
        }

        private void StartAudioDiagTimer()
        {
            if (_audioDiagTimer != null)
            {
                return;
            }
            _audioDiagTimer = new DispatcherTimer(DispatcherPriority.Background, _dispatcher)
            {
                Interval = TimeSpan.FromMilliseconds(200)
            };
            _audioDiagTimer.Tick += OnAudioDiagTick;
            _audioDiagTimer.Start();
        }

        private void StopAudioDiagTimer()
        {
            if (_audioDiagTimer == null)
            {
                return;
            }
            _audioDiagTimer.Stop();
            _audioDiagTimer.Tick -= OnAudioDiagTick;
            _audioDiagTimer = null;
        }

        private void OnAudioDiagTick(object? sender, EventArgs e)
        {
            IAudioOutput? output = _audioOutput;
            if (output == null)
            {
                return;
            }
            AudioBufferedMs = output.BufferedMs;
            AudioBufferLengthMs = output.BufferLengthMs;
            AudioFillRatio = output.FillRatio;
            AudioDropPercent = output.DropRatio * 100.0;
            AudioTotalDroppedMs = output.TotalDroppedMs;
        }

        private void OnSourceStatusChanged(object? sender, string message)
        {
            _dispatcher.BeginInvoke(() => StatusMessage = message);
        }

        private void OnFrameCapturedForDisplay(object? sender, RtspFrame frame)
        {
            _displayFpsCounter.Tick(out double fps);
            _dispatcher.BeginInvoke(() =>
            {
                RenderFrameToBitmap(frame);
                DisplayFps = fps;
            }, DispatcherPriority.Render);
        }

        private void RenderFrameToBitmap(RtspFrame frame)
        {
            if (_imageSource == null
                || _imageSource.PixelWidth != frame.Width
                || _imageSource.PixelHeight != frame.Height)
            {
                ImageSource = new WriteableBitmap(
                    frame.Width, frame.Height, 96, 96, PixelFormats.Bgr24, null);
                ImageWidth = frame.Width;
                ImageHeight = frame.Height;
            }

            int stride = frame.Width * 3;
            _imageSource!.WritePixels(
                new Int32Rect(0, 0, frame.Width, frame.Height),
                frame.BgrPixels,
                stride,
                0);
        }

        private async Task InferenceLoopAsync(CancellationToken ct)
        {
            if (_source == null)
            {
                return;
            }

            try
            {
                await foreach (RtspFrame frame in _source.Reader.ReadAllAsync(ct))
                {
                    if (ct.IsCancellationRequested)
                    {
                        break;
                    }

                    IRtspFrameDetector detector = _detectorResolver(_device);
                    var (detections, timings) = await detector
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
            }
            catch (Exception ex)
            {
                _ = _dispatcher.BeginInvoke(() => StatusMessage = $"추론 오류: {ex.Message}");
            }
        }

        public void Dispose()
        {
            StopAll();
        }

        private sealed class FpsCounter
        {
            private readonly Stopwatch _sw = Stopwatch.StartNew();
            private int _count;
            private long _lastReportMs;
            private double _lastFps;

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
                if (elapsed >= 1000)
                {
                    _lastFps = _count * 1000.0 / elapsed;
                    _count = 0;
                    _lastReportMs = _sw.ElapsedMilliseconds;
                }
                fps = _lastFps;
            }
        }
    }
}
