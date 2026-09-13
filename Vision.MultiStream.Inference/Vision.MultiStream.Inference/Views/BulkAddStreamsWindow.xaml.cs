using System;
using System.Collections.Generic;
using System.Windows;
using Vision.MultiStream.Inference.Services.Yolo;
using Vision.MultiStream.Inference.ViewModels;

namespace Vision.MultiStream.Inference.Views
{
    public partial class BulkAddStreamsWindow : Window
    {
        public BulkAddStreamsWindow()
        {
            InitializeComponent();

            // 현재 빌드(UseDirectML 토글)에서 적재되지 않는 디바이스 라디오는 비활성화.
            //   DirectML 빌드 → CUDA/TensorRT 비활성,  Gpu 빌드 → DML/NativeCpp 비활성
#if USE_DIRECTML
            GpuRadio.IsEnabled = false;
            TensorRtRadio.IsEnabled = false;
            TensorRtCudaPreRadio.IsEnabled = false;
#else
            DmlRadio.IsEnabled = false;
            NativeCppRadio.IsEnabled = false;
#endif
        }

        public InferenceDevice SelectedDevice { get; private set; } = InferenceDevice.Cpu;
        public bool SelectedInferenceEnabled { get; private set; } = true;
        public DecodeMode SelectedDecodeMode { get; private set; } = DecodeMode.Software;
        public DisplayMode SelectedDisplayMode { get; private set; } = DisplayMode.IndividualD3D;

        // 컴포지터가 없으면(GPU 없음) 컴포지터 표시기와 HW 디코딩(컴포지터 전용)을 비활성화. AddBulk 가 ShowDialog 전에 설정.
        public bool CompositorAvailable
        {
            get => DisplayCompositorRadio.IsEnabled;
            set
            {
                DisplayCompositorRadio.IsEnabled = value;
                DecodeHwRadio.IsEnabled = value;
            }
        }

        // HW 디코딩은 현재 컴포지터 표시기만 지원 → HW 선택 시 개별 표시기를 막고 컴포지터로 전환.
        // XAML 의 IsChecked="True" 가 InitializeComponent 중에 Checked 를 일으켜, 아직 생성 전인 라디오가 null 일 수 있다.
        private void OnDecodeModeChanged(object sender, RoutedEventArgs e)
        {
            if (DecodeHwRadio == null || DisplayBitmapRadio == null || DisplayD3DRadio == null || DisplayCompositorRadio == null)
            {
                return;
            }

            bool hardware = DecodeHwRadio.IsChecked == true;
            DisplayBitmapRadio.IsEnabled = !hardware;
            DisplayD3DRadio.IsEnabled = !hardware;
            if (hardware)
            {
                DisplayCompositorRadio.IsChecked = true;
            }
        }

        private string[] _urls = Array.Empty<string>();

        public IEnumerable<string> GetUrls() => _urls;

        private void OnAdd(object sender, RoutedEventArgs e)
        {
            _urls = (UrlsTextBox.Text ?? string.Empty)
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            if (DmlRadio.IsChecked == true)
            {
                SelectedDevice = InferenceDevice.DirectML;
            }
            else if (GpuRadio.IsChecked == true)
            {
                SelectedDevice = InferenceDevice.Gpu;
            }
            else if (TensorRtRadio.IsChecked == true)
            {
                SelectedDevice = InferenceDevice.TensorRT;
            }
            else if (TensorRtCudaPreRadio.IsChecked == true)
            {
                SelectedDevice = InferenceDevice.TensorRtCudaPre;
            }
            else if (NativeCppRadio.IsChecked == true)
            {
                SelectedDevice = InferenceDevice.NativeCpp;
            }
            else
            {
                SelectedDevice = InferenceDevice.Cpu;
            }

            SelectedInferenceEnabled = InferenceCheck.IsChecked == true;

            if (DecodeHwRadio.IsChecked == true)
            {
                SelectedDecodeMode = DecodeMode.Hardware;
            }
            else
            {
                SelectedDecodeMode = DecodeMode.Software;
            }

            if (DisplayCompositorRadio.IsChecked == true)
            {
                SelectedDisplayMode = DisplayMode.Compositor;
            }
            else if (DisplayBitmapRadio.IsChecked == true)
            {
                SelectedDisplayMode = DisplayMode.IndividualBitmap;
            }
            else
            {
                SelectedDisplayMode = DisplayMode.IndividualD3D;
            }

            DialogResult = true;
            Close();
        }

        private void OnCancel(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
