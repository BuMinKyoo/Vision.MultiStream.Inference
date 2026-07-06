using System;
using System.IO;
using System.Windows;
using Vision.MultiStream.Inference.Services.Direct3D;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vision.MultiStream.Inference.Services.Snapshot;
using Vision.MultiStream.Inference.Services.Yolo;
using Vision.MultiStream.Inference.ViewModels;

namespace Vision.MultiStream.Inference
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly YoloInferenceEngine? _cpuEngine;
        private readonly YoloInferenceEngine? _dmlEngine;
        private readonly YoloInferenceEngine? _gpuEngine;
        private readonly YoloInferenceEngine? _trtEngine;  // Phase 3.5 TensorRT
        private readonly YoloInferenceEngine? _trtCudaEngine; // Phase 4: TRT 추론 + CUDA 전/후처리
        private readonly NativeYoloEngine? _nativeEngine; // Phase 3 GPU(C++)
        private SnapshotViewModel? _snapshotVm;
        private MultiStreamViewModel? _multiStreamVm;
        private PerformanceViewModel? _performanceVm;

        private StreamCompositor? _compositor;

        public MainWindow()
        {
            InitializeComponent();

            // 윈도우 로드 후(HWND 확보 후) 컴포지터 생성. D3D9 실패(GPU/드라이버 부재 등) 시
            // 컴포지터 없이 진행 → 각 스트림은 기존 per-stream D3DImage 경로(이쪽도 실패하면
            // 추가로 WriteableBitmap CPU 경로) 로 자동 폴백.
            Loaded += (_, _) =>
            {
                try
                {
                    // surface 가 크면 셀별 다운스케일 비율이 줄어 YUV 샘플링 아티팩트(색감 시프트/깍두기) 감소.
                    // 단일 스트림(전체 surface 1타일) 이면 1:1 → 기존 per-stream presenter 화질과 동일.
                    var compositor = new StreamCompositor(2560, 1440);
                    compositor.Start();
                    _compositor = compositor;
                    MainCompositorImage.Source = compositor.Image;
                    _multiStreamVm?.AttachCompositor(compositor);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[Compositor] 초기화 실패 — per-stream 경로로 폴백: {ex.Message}");
                    _compositor?.Dispose();
                    _compositor = null;
                }
            };
            Closed += (_, _) => _compositor?.Dispose();

            string modelPath = Path.Combine(
                AppContext.BaseDirectory, "Assets", "Models", "yolov8n.onnx");
            // DirectML 전용 모델 경로. A/B 속도 비교 시 이 파일명만 yolov8n.onnx ↔ yolov8n_fp16.onnx 로 바꿔 리빌드한다.
            // (둘 다 입출력 FP32 + 형상 동일이라 전/후처리 코드는 그대로. CPU/TRT 폴백은 FP16 이득이 없으므로 modelPath(FP32) 유지.)
            string modelPathDml = Path.Combine(
                AppContext.BaseDirectory, "Assets", "Models", "yolov8n_fp16.onnx");

            if (!File.Exists(modelPath))
            {
                MessageBox.Show(
                    "ONNX 모델 파일을 찾을 수 없습니다.\n\n경로: " + modelPath +
                    "\n\nyolov8n.onnx 를 해당 경로에 배치하거나 빌드 후 다시 실행하세요.",
                    "모델 누락",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
                return;
            }

            try
            {
                _cpuEngine = new YoloInferenceEngine(modelPath, InferenceDevice.Cpu);
                // ORT 패키지(=UseDirectML 심볼)에 맞춰 사용 가능한 엔진만 생성. 나머지는 null → 선택 시 CPU 폴백.
#if USE_DIRECTML
                _dmlEngine = TryCreateEngine(modelPathDml, InferenceDevice.DirectML, "DirectML");
                // Phase 3 GPU(C++): 네이티브 DLL(vision_infer.dll)은 ORT+DirectML 전제 → DirectML 구성에서만 적재.
                _nativeEngine = TryCreateNativeEngine(modelPath);
                _gpuEngine = null;
                _trtEngine = null;
                _trtCudaEngine = null;
#else
                _dmlEngine = null;
                _nativeEngine = null;
                _gpuEngine = TryCreateEngine(modelPath, InferenceDevice.Gpu, "GPU(CUDA)");
                _trtEngine = TryCreateEngine(modelPath, InferenceDevice.TensorRT, "TensorRT");
                // TRT 추론 엔진을 하나 더(엔진 캐시는 디스크 공유라 재빌드 없음). 이쪽만 후처리를 CUDA 로.
                // 전처리는 detector 의 useCudaPreprocess 로 CUDA. → "CUDA전처리+TRT" 는 전/후처리 모두 GPU.
                _trtCudaEngine = TryCreateEngine(modelPath, InferenceDevice.TensorRT, "TensorRT(CUDA후처리)", useCudaPostprocess: true);
#endif

                var snapshotDetector = new SnapshotDetector(_cpuEngine);

                // 같은 ONNX 세션을 여러 스트림이 동시에 호출하지 않도록 디바이스별로 직렬화 래핑
                IRtspFrameDetector cpuRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_cpuEngine));
                // 현재 빌드에서 비활성인 디바이스(엔진 null)는 선택돼도 CPU 로 폴백.
                IRtspFrameDetector dmlRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_dmlEngine ?? _cpuEngine));
                IRtspFrameDetector gpuRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_gpuEngine ?? _cpuEngine));
                IRtspFrameDetector nativeRtspDetector = new SerializedFrameDetector(new RtspFrameDetector((IYoloEngine?)_nativeEngine ?? _cpuEngine));
                IRtspFrameDetector trtRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_trtEngine ?? _cpuEngine));
                // [Phase 4 Step 14] 추론은 위 TensorRT 엔진과 동일(공유, 엔진 내부 _runLock 이 동시성 보장),
                // 전처리만 CUDA 커널로 수행하는 조합. CPU 전처리 대비 효과 측정용.
                IRtspFrameDetector trtCudaPreRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_trtCudaEngine ?? _cpuEngine, useCudaPreprocess: true));

                _snapshotVm = new SnapshotViewModel(snapshotDetector);
                _multiStreamVm = new MultiStreamViewModel(cpuRtspDetector, dmlRtspDetector, gpuRtspDetector, nativeRtspDetector, trtRtspDetector, trtCudaPreRtspDetector);
                _performanceVm = new PerformanceViewModel();

                DataContext = new ShellViewModel(_snapshotVm, _multiStreamVm, _performanceVm);

                Closed += (_, _) =>
                {
                    _snapshotVm?.Dispose();
                    _multiStreamVm?.Dispose();
                    _performanceVm?.Dispose();
                    _cpuEngine?.Dispose();
                    _dmlEngine?.Dispose();
                    _gpuEngine?.Dispose();
                    _trtEngine?.Dispose();
                    _trtCudaEngine?.Dispose();
                    _nativeEngine?.Dispose();
                };

                // [Phase 3.5] VLM 모델을 백그라운드로 미리 적재 → 첫 사람 검출 시 콜드 로딩 지연 제거.
                // best-effort fire-and-forget (Ollama 미기동 시 내부에서 조용히 무시).
                _ = StreamItemViewModel.WarmUpVlmAsync();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"초기화 실패:\n{ex.Message}",
                    "초기화 오류",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }
        }

        private YoloInferenceEngine? TryCreateEngine(string modelPath, InferenceDevice device, string label, bool useCudaPostprocess = false)
        {
            try
            {
                return new YoloInferenceEngine(modelPath, device, useCudaPostprocess);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"{label} 초기화 실패 - 해당 옵션은 CPU로 폴백됩니다.\n\n{ex.Message}",
                    $"{label} 경고",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning);
                return null;
            }
        }

        // Phase 3 GPU(C++): 네이티브 DLL(vision_infer.dll) 엔진. DLL 누락/DML 실패 시 null → CPU 폴백.
        private NativeYoloEngine? TryCreateNativeEngine(string modelPath)
        {
            try
            {
                return new NativeYoloEngine(modelPath);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"GPU(C++) 초기화 실패 - 해당 옵션은 CPU로 폴백됩니다.\n\n{ex.Message}",
                    "GPU(C++) 경고",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning);
                return null;
            }
        }
    }
}
