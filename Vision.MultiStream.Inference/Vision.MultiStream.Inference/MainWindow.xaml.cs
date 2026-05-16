using System;
using System.IO;
using System.Windows;
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
        private SnapshotViewModel? _snapshotVm;
        private MultiStreamViewModel? _multiStreamVm;
        private Mp3PlayerViewModel? _mp3PlayerVm;
        private PerformanceViewModel? _performanceVm;

        public MainWindow()
        {
            InitializeComponent();

            string modelPath = Path.Combine(
                AppContext.BaseDirectory, "Assets", "Models", "yolov8n.onnx");

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
                _dmlEngine = TryCreateEngine(modelPath, InferenceDevice.DirectML, "DirectML");
                _gpuEngine = TryCreateEngine(modelPath, InferenceDevice.Gpu, "GPU(CUDA)");

                var snapshotDetector = new SnapshotDetector(_cpuEngine);

                // 같은 ONNX 세션을 여러 스트림이 동시에 호출하지 않도록 디바이스별로 직렬화 래핑
                IRtspFrameDetector cpuRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_cpuEngine));
                IRtspFrameDetector dmlRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_dmlEngine ?? _cpuEngine));
                IRtspFrameDetector gpuRtspDetector = new SerializedFrameDetector(new RtspFrameDetector(_gpuEngine ?? _cpuEngine));

                _snapshotVm = new SnapshotViewModel(snapshotDetector);
                _multiStreamVm = new MultiStreamViewModel(cpuRtspDetector, dmlRtspDetector, gpuRtspDetector);
                _mp3PlayerVm = new Mp3PlayerViewModel();
                _performanceVm = new PerformanceViewModel();

                DataContext = new ShellViewModel(_snapshotVm, _multiStreamVm, _mp3PlayerVm, _performanceVm);

                Closed += (_, _) =>
                {
                    _snapshotVm?.Dispose();
                    _multiStreamVm?.Dispose();
                    _mp3PlayerVm?.Dispose();
                    _performanceVm?.Dispose();
                    _cpuEngine?.Dispose();
                    _dmlEngine?.Dispose();
                    _gpuEngine?.Dispose();
                };
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

        private YoloInferenceEngine? TryCreateEngine(string modelPath, InferenceDevice device, string label)
        {
            try
            {
                return new YoloInferenceEngine(modelPath, device);
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
    }
}
