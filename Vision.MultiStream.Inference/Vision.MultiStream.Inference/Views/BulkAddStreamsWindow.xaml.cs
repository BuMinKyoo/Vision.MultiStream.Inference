using System;
using System.Collections.Generic;
using System.Windows;
using Vision.MultiStream.Inference.Services.Yolo;

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
        public bool SelectedUseHardwareDecoding { get; private set; }

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
            SelectedUseHardwareDecoding = HwDecoderRadio.IsChecked == true;

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
