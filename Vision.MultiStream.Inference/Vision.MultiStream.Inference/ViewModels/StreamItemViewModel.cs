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
    /// 영상/소리 두 개의 독립 토글:
    ///   - IsVideoEnabled : 비디오 디코딩 + 표시 + 추론
    ///   - IsAudioEnabled : 오디오 디코딩 + 스피커 출력
    /// 두 토글은 독립적으로 ON/OFF 가능. 둘 다 OFF 면 RTSP 연결 자체를 끊음.
    /// 한쪽만 토글하면 RTSP 연결을 잠깐 재수립한다 (간단함을 위한 의도적 선택).
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
        private bool _isAudioEnabled;
        private string _statusMessage = "대기";
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

        public StreamItemViewModel(
            string name,
            string rtspUrl,
            InferenceDevice device,
            Func<InferenceDevice, IRtspFrameDetector> detectorResolver,
            Action<StreamItemViewModel> onRemoveRequested)
        {
            _name = name;
            _rtspUrl = rtspUrl;
            _device = device;
            _detectorResolver = detectorResolver;
            _onRemoveRequested = onRemoveRequested;
            _dispatcher = Application.Current.Dispatcher;

            ToggleVideoCommand = new RelayCommand(ToggleVideo, () => !string.IsNullOrWhiteSpace(RtspUrl));
            ToggleAudioCommand = new RelayCommand(ToggleAudio, () => !string.IsNullOrWhiteSpace(RtspUrl));
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
                ToggleAudioCommand.RaiseCanExecuteChanged();
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

        // 외부(MultiStreamViewModel 의 전체 ON/OFF) 에서 사용할 수 있는 직접 setter
        public void SetVideo(bool enabled)
        {
            if (_isVideoEnabled == enabled)
            {
                return;
            }
            _isVideoEnabled = enabled;
            OnPropertyChanged(nameof(IsVideoEnabled));
            OnPropertyChanged(nameof(IsActive));
            StopCommand.RaiseCanExecuteChanged();
            ApplyState();
        }

        public void SetAudio(bool enabled)
        {
            if (_isAudioEnabled == enabled)
            {
                return;
            }
            _isAudioEnabled = enabled;
            OnPropertyChanged(nameof(IsAudioEnabled));
            OnPropertyChanged(nameof(IsActive));
            StopCommand.RaiseCanExecuteChanged();
            ApplyState();
        }

        private void ToggleVideo()
        {
            SetVideo(!_isVideoEnabled);
        }

        private void ToggleAudio()
        {
            SetAudio(!_isAudioEnabled);
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
                ApplyState();
            }
        }

        /// <summary>
        /// 두 토글의 현재 상태에 맞춰 source 를 재구성한다.
        /// 어느 한쪽이라도 토글되면 항상 Stop → Start 로 단순화 (의도적인 짧은 reconnect).
        /// </summary>
        private void ApplyState()
        {
            // 항상 기존 source 닫고 새로 구성
            TearDownSource();

            bool wantVideo = _isVideoEnabled;
            bool wantAudio = _isAudioEnabled;

            if (!wantVideo && !wantAudio)
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
                return;
            }

            try
            {
                _displayFpsCounter.Reset();
                _inferenceFpsCounter.Reset();

                _source = new RtspFrameSource(RtspUrl);
                _source.StatusChanged += OnSourceStatusChanged;
                _source.FrameCaptured += OnFrameCapturedForDisplay;

                IAudioOutput? audioOutput = wantAudio ? new WasapiAudioOutput() : null;
                _source.Start(wantVideo, audioOutput);

                if (wantVideo)
                {
                    _cts = new CancellationTokenSource();
                    _inferenceTask = Task.Run(() => InferenceLoopAsync(_cts.Token));
                }

                StatusMessage = "연결 중...";
            }
            catch (Exception ex)
            {
                StatusMessage = $"연결 실패: {ex.Message}";
                TearDownSource();
            }
        }

        private void TearDownSource()
        {
            try
            {
                _cts?.Cancel();

                if (_source != null)
                {
                    _source.FrameCaptured -= OnFrameCapturedForDisplay;
                    _source.StatusChanged -= OnSourceStatusChanged;
                    _source.Stop();
                    _source.Dispose();
                    _source = null;
                }

                _inferenceTask = null;
                _cts?.Dispose();
                _cts = null;
            }
            catch
            {
                // teardown 중 예외는 무시
            }
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
