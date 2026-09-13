using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Windows;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Services.Direct3D;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vision.MultiStream.Inference.Services.Vlm;
using Vision.MultiStream.Inference.Services.Yolo;
using Vision.MultiStream.Inference.Views;

namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 스트림 표시/디코딩 방식.
    ///   - CpuIndividual       : SW 디코딩 + 개별 표시, D3D9 셰이더로 그리기(D3DImageYuvPresenter).
    ///   - CpuIndividualBitmap : SW 디코딩 + 개별 표시, WriteableBitmap 으로 CPU 그리기(D3D 불필요).
    ///   - CpuCompositor : SW 디코딩 + 단일 컴포지터 표시.
    ///   - GpuCompositor : HW(D3D11VA) 디코딩 + 단일 컴포지터 표시.
    /// 컴포지터 두 모드는 GPU/컴포지터가 있어야 선택 가능하다.
    /// </summary>
    public enum StreamRenderMode { CpuIndividual, CpuIndividualBitmap, CpuCompositor, GpuCompositor }

    /// <summary>
    /// 다중 RTSP 스트림 관리 ViewModel.
    /// 책임: StreamItemViewModel 컬렉션 + 사이드 패널의 추가/삭제/일괄/전체시작중지 명령 + 컴포지터 타일 배치.
    /// 추론 디바이스(IRtspFrameDetector)는 이미 SerializedFrameDetector로 래핑된 상태로 주입받음.
    /// </summary>
    public sealed class MultiStreamViewModel : BaseViewModel, IDisposable
    {
        private readonly Func<InferenceDevice, IRtspFrameDetector> _detectorResolver;

        private string _newName = string.Empty;
        private string _newRtspUrl = "rtsp://localhost:8554/cam1";
        private InferenceDevice _newDevice = InferenceDevice.Cpu;
        private bool _newUseInference = true;
        // 컴포지터가 없으면(GPU 없음) 기본값은 개별. AttachCompositor 시 컴포지터 모드로 올린다.
        private StreamRenderMode _newRenderMode = StreamRenderMode.CpuIndividual;
        private int _autoCounter = 1;
        private StreamCompositor? _compositor;

        public MultiStreamViewModel(
            IRtspFrameDetector cpuDetector,
            IRtspFrameDetector dmlDetector,
            IRtspFrameDetector gpuDetector,
            IRtspFrameDetector nativeDetector,
            IRtspFrameDetector trtDetector,
            IRtspFrameDetector trtCudaPreDetector)
        {
            _detectorResolver = device => device switch
            {
                InferenceDevice.DirectML => dmlDetector,
                InferenceDevice.Gpu => gpuDetector,
                InferenceDevice.NativeCpp => nativeDetector,
                InferenceDevice.TensorRT => trtDetector,
                InferenceDevice.TensorRtCudaPre => trtCudaPreDetector,
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
            StartAllVlmCommand = new RelayCommand(() => SetAllVlm(true));
            StopAllVlmCommand = new RelayCommand(() => SetAllVlm(false));
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

            // 컴포지터가 생겼으니 컴포지터 라디오를 활성화한다(기본 선택값은 바꾸지 않음).
            OnPropertyChanged(nameof(IsCompositorAvailable));

            foreach (var s in Streams)
            {
                WireStream(s);
            }
            RecomputeLayout();
        }

        // 컴포지터가 만들어졌는지(=CPU+컴포지터 / GPU+컴포지터 선택 가능 여부). 라디오 IsEnabled 바인딩용.
        public bool IsCompositorAvailable => _compositor != null;

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

        // WPF UniformGrid 자동 분할 규칙(cols = rows = ceil(sqrt(N)))을 그대로 따라
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

            // WPF UniformGrid 의 실제 자동 분할 규칙과 동일하게 맞춤
            // (cols=rows=ceil(sqrt(N)) → 정사각 격자. N=2 면 2×2 = 4셀, 위 2개만 채움).
            int cols = (int)Math.Ceiling(Math.Sqrt(n));
            int rows = cols;
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
        public RelayCommand StartAllVlmCommand { get; }
        public RelayCommand StopAllVlmCommand { get; }
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
                OnPropertyChanged(nameof(NewUseNativeCpp));
                OnPropertyChanged(nameof(NewUseTensorRT));
                OnPropertyChanged(nameof(NewUseTensorRtCudaPre));
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

        public bool NewUseNativeCpp
        {
            get => _newDevice == InferenceDevice.NativeCpp;
            set
            {
                if (value)
                {
                    NewDevice = InferenceDevice.NativeCpp;
                }
            }
        }

        public bool NewUseTensorRT
        {
            get => _newDevice == InferenceDevice.TensorRT;
            set
            {
                if (value)
                {
                    NewDevice = InferenceDevice.TensorRT;
                }
            }
        }

        public bool NewUseTensorRtCudaPre
        {
            get => _newDevice == InferenceDevice.TensorRtCudaPre;
            set
            {
                if (value)
                {
                    NewDevice = InferenceDevice.TensorRtCudaPre;
                }
            }
        }

        // 현재 빌드(UseDirectML 토글)에서 실제로 적재된 엔진만 라디오 IsEnabled 로 노출.
        //   DirectML 빌드 → CPU/DML/NativeCpp 사용, CUDA/TensorRT 비활성
        //   Gpu 빌드      → CPU/CUDA/TensorRT 사용, DML/NativeCpp 비활성
        // 한 빌드에 양쪽 EP 공존이 불가하므로 컴파일 심볼로 결정한다(런타임 변동 없음 → 상수 게터).
#if USE_DIRECTML
        public bool IsDirectMLAvailable => true;
        public bool IsNativeCppAvailable => true;
        public bool IsCudaAvailable => false;
        public bool IsTensorRTAvailable => false;
        public bool IsTensorRtCudaPreAvailable => false;
#else
        public bool IsDirectMLAvailable => false;
        public bool IsNativeCppAvailable => false;
        public bool IsCudaAvailable => true;
        public bool IsTensorRTAvailable => true;
        // CUDA 전처리 커널(vision_cuda.dll)은 Gpu 빌드에서만 빌드/복사되므로 TensorRT 와 동일 가용성.
        public bool IsTensorRtCudaPreAvailable => true;
#endif

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

        // 표시/디코딩 모드 선택 라디오 3종. 컴포지터 두 모드는 IsCompositorAvailable 일 때만 켤 수 있다.
        public StreamRenderMode NewRenderMode
        {
            get => _newRenderMode;
            set
            {
                if (_newRenderMode == value)
                {
                    return;
                }
                _newRenderMode = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(NewModeCpuIndividual));
                OnPropertyChanged(nameof(NewModeCpuIndividualBitmap));
                OnPropertyChanged(nameof(NewModeCpuCompositor));
                OnPropertyChanged(nameof(NewModeGpuCompositor));
            }
        }

        public bool NewModeCpuIndividual
        {
            get => _newRenderMode == StreamRenderMode.CpuIndividual;
            set
            {
                if (value)
                {
                    NewRenderMode = StreamRenderMode.CpuIndividual;
                }
            }
        }

        public bool NewModeCpuIndividualBitmap
        {
            get => _newRenderMode == StreamRenderMode.CpuIndividualBitmap;
            set
            {
                if (value)
                {
                    NewRenderMode = StreamRenderMode.CpuIndividualBitmap;
                }
            }
        }

        public bool NewModeCpuCompositor
        {
            get => _newRenderMode == StreamRenderMode.CpuCompositor;
            set
            {
                if (value)
                {
                    NewRenderMode = StreamRenderMode.CpuCompositor;
                }
            }
        }

        public bool NewModeGpuCompositor
        {
            get => _newRenderMode == StreamRenderMode.GpuCompositor;
            set
            {
                if (value)
                {
                    NewRenderMode = StreamRenderMode.GpuCompositor;
                }
            }
        }

        // [Step 7] VLM 묘사 디바이스 전역 토글(GPU/CPU). Ollama 단일 모델 로드라 앱 공통.
        // 상단 툴바 라디오 2개(GPU/CPU)에 바인딩. 전 스트림 공통, 다음 VLM 호출부터 적용(모델 재로딩 ~1분 발생 가능).
        public bool VlmUseGpu
        {
            get => OllamaVlmClient.UseGpu;
            set
            {
                if (OllamaVlmClient.UseGpu == value)
                {
                    return;
                }
                OllamaVlmClient.UseGpu = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(VlmUseCpu));
            }
        }

        public bool VlmUseCpu
        {
            get => !OllamaVlmClient.UseGpu;
            set
            {
                if (value)
                {
                    VlmUseGpu = false;
                }
            }
        }

        public int TotalCount => Streams.Count;

        private void AddStream()
        {
            string name = string.IsNullOrWhiteSpace(NewName) ? $"cam{_autoCounter++}" : NewName.Trim();
            var item = CreateStream(name, NewRtspUrl.Trim(), NewDevice, _newUseInference, _newRenderMode);
            Streams.Add(item);
            NewName = string.Empty;
            OnPropertyChanged(nameof(TotalCount));
        }

        private void AddBulk()
        {
            var dialog = new BulkAddStreamsWindow
            {
                Owner = Application.Current?.MainWindow,
                CompositorAvailable = IsCompositorAvailable
            };

            if (dialog.ShowDialog() != true)
            {
                return;
            }

            InferenceDevice device = dialog.SelectedDevice;
            bool inferenceEnabled = dialog.SelectedInferenceEnabled;
            StreamRenderMode renderMode = dialog.SelectedRenderMode;
            foreach (string url in dialog.GetUrls())
            {
                string trimmed = url.Trim();
                if (string.IsNullOrEmpty(trimmed))
                {
                    continue;
                }
                Streams.Add(CreateStream($"cam{_autoCounter++}", trimmed, device, inferenceEnabled, renderMode));
            }
            OnPropertyChanged(nameof(TotalCount));
        }

        // VLM 은 부하가 크므로 기본 OFF 로 추가하고, 필요한 스트림만 💬 토글(또는 전체 VLM ON)로 켠다.
        private StreamItemViewModel CreateStream(string name, string url, InferenceDevice device, bool inferenceEnabled, StreamRenderMode renderMode)
        {
            var item = new StreamItemViewModel(name, url, device, _detectorResolver, OnRemoveRequested, inferenceEnabled, renderMode, initialVlmEnabled: false);
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

        private void SetAllVlm(bool enabled)
        {
            foreach (var s in Streams)
            {
                s.SetVlm(enabled);
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
