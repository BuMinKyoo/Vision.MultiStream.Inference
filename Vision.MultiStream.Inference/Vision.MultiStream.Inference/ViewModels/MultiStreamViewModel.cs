using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Windows;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Services.Direct3D;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vision.MultiStream.Inference.Services.Yolo;
using Vision.MultiStream.Inference.Views;

namespace Vision.MultiStream.Inference.ViewModels
{
    public enum LayoutMode { Auto, Grid2x2, Grid3x3, Grid4x4 }

    /// <summary>
    /// 다중 RTSP 스트림 관리 ViewModel.
    /// 책임: StreamItemViewModel 컬렉션 + 사이드 패널의 추가/삭제/일괄/전체시작중지/레이아웃 명령.
    /// 추론 디바이스(IRtspFrameDetector)는 이미 SerializedFrameDetector로 래핑된 상태로 주입받음.
    /// </summary>
    public sealed class MultiStreamViewModel : BaseViewModel, IDisposable
    {
        private readonly Func<InferenceDevice, IRtspFrameDetector> _detectorResolver;

        private string _newName = string.Empty;
        private string _newRtspUrl = "rtsp://localhost:8554/cam1";
        private InferenceDevice _newDevice = InferenceDevice.Cpu;
        private bool _newUseInference = true;
        private LayoutMode _layout = LayoutMode.Auto;
        private int _autoCounter = 1;
        private StreamCompositor? _compositor;

        public MultiStreamViewModel(
            IRtspFrameDetector cpuDetector,
            IRtspFrameDetector dmlDetector,
            IRtspFrameDetector gpuDetector)
        {
            _detectorResolver = device => device switch
            {
                InferenceDevice.DirectML => dmlDetector,
                InferenceDevice.Gpu => gpuDetector,
                _ => cpuDetector
            };

            AddStreamCommand = new RelayCommand(AddStream, () => !string.IsNullOrWhiteSpace(NewRtspUrl));
            AddBulkCommand = new RelayCommand(AddBulk);
            StartAllVideoCommand = new RelayCommand(() => SetAllVideo(true));
            StopAllVideoCommand = new RelayCommand(() => SetAllVideo(false));
            StartAllAudioCommand = new RelayCommand(() => SetAllAudio(true));
            StopAllAudioCommand = new RelayCommand(() => SetAllAudio(false));
            StartAllInferenceCommand = new RelayCommand(() => SetAllInference(true));
            StopAllInferenceCommand = new RelayCommand(() => SetAllInference(false));
            StartAllCommand = new RelayCommand(StartAll);
            StopAllCommand = new RelayCommand(StopAll);
            RemoveAllCommand = new RelayCommand(RemoveAll);

            // 스트림 추가/삭제 시 레이아웃 재계산.
            Streams.CollectionChanged += OnStreamsCollectionChanged;
        }

        private void OnStreamsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            RecomputeLayout();
        }

        public ObservableCollection<StreamItemViewModel> Streams { get; } = new();

        // 윈도우 로드 후 MainWindow 가 컴포지터를 주입. 기존+이후 스트림에 전파.
        public void AttachCompositor(StreamCompositor compositor)
        {
            _compositor = compositor;
            foreach (var s in Streams)
            {
                WireStream(s);
            }
            RecomputeLayout();
        }

        // 스트림 추가 시 컴포지터 연결 + 슬롯 변경 이벤트 구독.
        private void WireStream(StreamItemViewModel item)
        {
            item.CompositorSlotChanged += OnAnySlotChanged;
            if (_compositor != null)
            {
                item.AttachCompositor(_compositor);
            }
        }

        private void UnwireStream(StreamItemViewModel item)
        {
            item.CompositorSlotChanged -= OnAnySlotChanged;
        }

        private void OnAnySlotChanged()
        {
            RecomputeLayout();
        }

        // WPF UniformGrid 규칙(Auto = ceil(sqrt(N)), Grid2x2/3x3/4x4 = 그 열수)을 그대로 따라
        // 컴포지터 surface 좌표계로 셀 rect 를 계산해 SetLayout 으로 전달.
        // 슬롯이 없는(시작 안 한) 스트림은 빈 자리로 두고 그 셀은 컴포지터에서 검정으로 남는다.
        private void RecomputeLayout()
        {
            if (_compositor == null)
            {
                return;
            }
            int n = Streams.Count;
            if (n == 0)
            {
                _compositor.SetLayout(Array.Empty<TileRect>());
                return;
            }

            int cols, rows;
            if (GridColumns > 0)
            {
                // 고정 그리드 모드(Grid2x2/3x3/4x4): rows 도 같이 고정 → 스트림이 모자라도 같은 격자 유지.
                cols = GridColumns;
                rows = GridRows;
            }
            else
            {
                // Auto: WPF UniformGrid 의 실제 자동 분할 규칙과 동일하게 맞춤
                // (cols=rows=ceil(sqrt(N)) → 정사각 격자. N=2 면 2×2 = 4셀, 위 2개만 채움).
                cols = (int)Math.Ceiling(Math.Sqrt(n));
                rows = cols;
            }
            int cellW = _compositor.Width / cols;
            int cellH = _compositor.Height / rows;

            var rects = new List<TileRect>(n);
            for (int i = 0; i < n; i++)
            {
                StreamItemViewModel item = Streams[i];
                if (item.CompositorSlotId < 0)
                {
                    continue;
                }
                int col = i % cols;
                int row = i / cols;
                rects.Add(new TileRect(item.CompositorSlotId, col * cellW, row * cellH, cellW, cellH));
            }
            _compositor.SetLayout(rects);
        }

        public RelayCommand AddStreamCommand { get; }
        public RelayCommand AddBulkCommand { get; }
        public RelayCommand StartAllVideoCommand { get; }
        public RelayCommand StopAllVideoCommand { get; }
        public RelayCommand StartAllAudioCommand { get; }
        public RelayCommand StopAllAudioCommand { get; }
        public RelayCommand StartAllInferenceCommand { get; }
        public RelayCommand StopAllInferenceCommand { get; }
        public RelayCommand StartAllCommand { get; }
        public RelayCommand StopAllCommand { get; }
        public RelayCommand RemoveAllCommand { get; }

        public string NewName
        {
            get => _newName;
            set
            {
                if (_newName != value)
                {
                    _newName = value;
                    OnPropertyChanged();
                }
            }
        }

        public string NewRtspUrl
        {
            get => _newRtspUrl;
            set
            {
                if (_newRtspUrl == value)
                {
                    return;
                }
                _newRtspUrl = value;
                OnPropertyChanged();
                AddStreamCommand.RaiseCanExecuteChanged();
            }
        }

        public InferenceDevice NewDevice
        {
            get => _newDevice;
            set
            {
                if (_newDevice == value)
                {
                    return;
                }
                _newDevice = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(NewUseCpu));
                OnPropertyChanged(nameof(NewUseDirectML));
                OnPropertyChanged(nameof(NewUseGpu));
            }
        }

        public bool NewUseCpu
        {
            get => _newDevice == InferenceDevice.Cpu;
            set
            {
                if (value)
                {
                    NewDevice = InferenceDevice.Cpu;
                }
            }
        }

        public bool NewUseDirectML
        {
            get => _newDevice == InferenceDevice.DirectML;
            set
            {
                if (value)
                {
                    NewDevice = InferenceDevice.DirectML;
                }
            }
        }

        public bool NewUseGpu
        {
            get => _newDevice == InferenceDevice.Gpu;
            set
            {
                if (value)
                {
                    NewDevice = InferenceDevice.Gpu;
                }
            }
        }

        public bool NewUseInference
        {
            get => _newUseInference;
            set
            {
                if (_newUseInference == value)
                {
                    return;
                }
                _newUseInference = value;
                OnPropertyChanged();
            }
        }

        public LayoutMode Layout
        {
            get => _layout;
            set
            {
                if (_layout == value)
                {
                    return;
                }
                _layout = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(GridColumns));
                OnPropertyChanged(nameof(GridRows));
                RecomputeLayout();
            }
        }

        // UniformGrid.Columns 바인딩용. 0 = 자동 (UniformGrid 기본 동작).
        public int GridColumns => _layout switch
        {
            LayoutMode.Grid2x2 => 2,
            LayoutMode.Grid3x3 => 3,
            LayoutMode.Grid4x4 => 4,
            _ => 0
        };

        // UniformGrid.Rows 바인딩용. 0 = 자동. Grid2x2/3x3/4x4 는 rows 도 고정해서
        // 스트림 수가 모자라도 surface 가 같은 격자로 분할되도록 한다.
        public int GridRows => _layout switch
        {
            LayoutMode.Grid2x2 => 2,
            LayoutMode.Grid3x3 => 3,
            LayoutMode.Grid4x4 => 4,
            _ => 0
        };

        public int TotalCount => Streams.Count;

        private void AddStream()
        {
            string name = string.IsNullOrWhiteSpace(NewName) ? $"cam{_autoCounter++}" : NewName.Trim();
            var item = CreateStream(name, NewRtspUrl.Trim(), NewDevice, _newUseInference);
            Streams.Add(item);
            NewName = string.Empty;
            OnPropertyChanged(nameof(TotalCount));
        }

        private void AddBulk()
        {
            var dialog = new BulkAddStreamsWindow
            {
                Owner = Application.Current?.MainWindow
            };

            if (dialog.ShowDialog() != true)
            {
                return;
            }

            InferenceDevice device = dialog.SelectedDevice;
            bool inferenceEnabled = dialog.SelectedInferenceEnabled;
            foreach (string url in dialog.GetUrls())
            {
                string trimmed = url.Trim();
                if (string.IsNullOrEmpty(trimmed))
                {
                    continue;
                }
                Streams.Add(CreateStream($"cam{_autoCounter++}", trimmed, device, inferenceEnabled));
            }
            OnPropertyChanged(nameof(TotalCount));
        }

        private StreamItemViewModel CreateStream(string name, string url, InferenceDevice device, bool inferenceEnabled)
        {
            var item = new StreamItemViewModel(name, url, device, _detectorResolver, OnRemoveRequested, inferenceEnabled);
            WireStream(item);
            return item;
        }

        private void OnRemoveRequested(StreamItemViewModel item)
        {
            UnwireStream(item);
            item.Dispose();
            Streams.Remove(item);
            OnPropertyChanged(nameof(TotalCount));
        }

        private void SetAllVideo(bool enabled)
        {
            foreach (var s in Streams)
            {
                s.SetVideo(enabled);
            }
        }

        private void SetAllAudio(bool enabled)
        {
            foreach (var s in Streams)
            {
                s.SetAudio(enabled);
            }
        }

        private void SetAllInference(bool enabled)
        {
            foreach (var s in Streams)
            {
                s.SetInference(enabled);
            }
        }

        private void StartAll()
        {
            foreach (var s in Streams)
            {
                s.SetVideo(true);
                s.SetAudio(true);
            }
        }

        private void StopAll()
        {
            foreach (var s in Streams)
            {
                s.SetVideo(false);
                s.SetAudio(false);
            }
        }

        private void RemoveAll()
        {
            foreach (var s in Streams)
            {
                UnwireStream(s);
                s.Dispose();
            }
            Streams.Clear();
            OnPropertyChanged(nameof(TotalCount));
        }

        public void Dispose()
        {
            Streams.CollectionChanged -= OnStreamsCollectionChanged;
            foreach (var s in Streams)
            {
                UnwireStream(s);
                s.Dispose();
            }
            Streams.Clear();
        }
    }
}
