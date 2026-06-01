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
        }

        public InferenceDevice SelectedDevice { get; private set; } = InferenceDevice.Cpu;
        public bool SelectedInferenceEnabled { get; private set; } = true;
        public StreamRenderMode SelectedRenderMode { get; private set; } = StreamRenderMode.CpuIndividual;

        // 컴포지터가 없으면(GPU 없음) 컴포지터 모드 라디오를 비활성화. AddBulk 가 ShowDialog 전에 설정.
        public bool CompositorAvailable
        {
            get => CpuCompositorRadio.IsEnabled;
            set
            {
                CpuCompositorRadio.IsEnabled = value;
                GpuCompositorRadio.IsEnabled = value;
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
            else
            {
                SelectedDevice = InferenceDevice.Cpu;
            }

            SelectedInferenceEnabled = InferenceCheck.IsChecked == true;

            if (GpuCompositorRadio.IsChecked == true)
            {
                SelectedRenderMode = StreamRenderMode.GpuCompositor;
            }
            else if (CpuCompositorRadio.IsChecked == true)
            {
                SelectedRenderMode = StreamRenderMode.CpuCompositor;
            }
            else
            {
                SelectedRenderMode = StreamRenderMode.CpuIndividual;
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
