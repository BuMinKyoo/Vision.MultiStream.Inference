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
    /// 멀티스트림에서 1개 RTSP 스트림을 표현하는 ViewModel.
    /// 책임: 자기 자신의 RtspFrameSource + 추론 루프 수명 관리.
    /// 공유 자원(디바이스별 IRtspFrameDetector)는 외부에서 주입받음.
    /// </summary>
    public sealed class StreamItemViewModel : BaseViewModel, IDisposable
    {
        private readonly Func<InferenceDevice, IRtspFrameDetector> _detectorResolver;
        private readonly Action<StreamItemViewModel> _onRemoveRequested;
        private readonly Dispatcher _dispatcher;

        private string _name;
        private string _rtspUrl;
        private InferenceDevice _device;
        private bool _isActive;
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

            StartCommand = new RelayCommand(Start, () => !IsActive && !string.IsNullOrWhiteSpace(RtspUrl));
            StopCommand = new RelayCommand(Stop, () => IsActive);
            ToggleCommand = new RelayCommand(() =>
            {
                if (IsActive)
                {
                    Stop();
                }
                else
                {
                    Start();
                }
            });
            RemoveCommand = new RelayCommand(() =>
            {
                Stop();
                _onRemoveRequested(this);
            });
        }

        public ObservableCollection<Detection> Detections { get; } = new();

        public RelayCommand StartCommand { get; }
        public RelayCommand StopCommand { get; }
        public RelayCommand ToggleCommand { get; }
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
                StartCommand.RaiseCanExecuteChanged();
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

        public bool IsActive
        {
            get => _isActive;
            private set
            {
                if (_isActive == value)
                {
                    return;
                }
                _isActive = value;
                OnPropertyChanged();
                StartCommand.RaiseCanExecuteChanged();
                StopCommand.RaiseCanExecuteChanged();
            }
        }

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

        public void Start()
        {
            if (IsActive)
            {
                return;
            }
            if (string.IsNullOrWhiteSpace(RtspUrl))
            {
                return;
            }

            try
            {
                _displayFpsCounter.Reset();
                _inferenceFpsCounter.Reset();

                _source = new RtspFrameSource(RtspUrl);
                _source.StatusChanged += OnSourceStatusChanged;
                _source.FrameCaptured += OnFrameCapturedForDisplay;
                _source.Start();

                _cts = new CancellationTokenSource();
                _inferenceTask = Task.Run(() => InferenceLoopAsync(_cts.Token));

                IsActive = true;
                StatusMessage = "연결 중...";
            }
            catch (Exception ex)
            {
                StatusMessage = $"연결 실패: {ex.Message}";
                Stop();
            }
        }

        public void Stop()
        {
            if (!IsActive && _source == null)
            {
                return;
            }

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

                IsActive = false;
                Detections.Clear();
                DisplayFps = 0;
                InferenceFps = 0;
                PreprocessMs = 0;
                InferenceMs = 0;
                PostprocessMs = 0;
                StatusMessage = "정지";
            }
            catch (Exception ex)
            {
                StatusMessage = $"중지 중 오류: {ex.Message}";
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
            Stop();
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
