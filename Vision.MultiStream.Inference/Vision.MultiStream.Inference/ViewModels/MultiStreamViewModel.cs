using System;
using System.Collections.ObjectModel;
using System.Windows;
using Vision.MultiStream.Inference.Common;
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
        private LayoutMode _layout = LayoutMode.Auto;
        private int _autoCounter = 1;

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
            StartAllCommand = new RelayCommand(StartAll);
            StopAllCommand = new RelayCommand(StopAll);
            RemoveAllCommand = new RelayCommand(RemoveAll);
        }

        public ObservableCollection<StreamItemViewModel> Streams { get; } = new();

        public RelayCommand AddStreamCommand { get; }
        public RelayCommand AddBulkCommand { get; }
        public RelayCommand StartAllVideoCommand { get; }
        public RelayCommand StopAllVideoCommand { get; }
        public RelayCommand StartAllAudioCommand { get; }
        public RelayCommand StopAllAudioCommand { get; }
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

        public int TotalCount => Streams.Count;

        private void AddStream()
        {
            string name = string.IsNullOrWhiteSpace(NewName) ? $"cam{_autoCounter++}" : NewName.Trim();
            var item = CreateStream(name, NewRtspUrl.Trim(), NewDevice);
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
            foreach (string url in dialog.GetUrls())
            {
                string trimmed = url.Trim();
                if (string.IsNullOrEmpty(trimmed))
                {
                    continue;
                }
                Streams.Add(CreateStream($"cam{_autoCounter++}", trimmed, device));
            }
            OnPropertyChanged(nameof(TotalCount));
        }

        private StreamItemViewModel CreateStream(string name, string url, InferenceDevice device)
        {
            return new StreamItemViewModel(name, url, device, _detectorResolver, OnRemoveRequested);
        }

        private void OnRemoveRequested(StreamItemViewModel item)
        {
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
                s.Dispose();
            }
            Streams.Clear();
            OnPropertyChanged(nameof(TotalCount));
        }

        public void Dispose()
        {
            foreach (var s in Streams)
            {
                s.Dispose();
            }
            Streams.Clear();
        }
    }
}
