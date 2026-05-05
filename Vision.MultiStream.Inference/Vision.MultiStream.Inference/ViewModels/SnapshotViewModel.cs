using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Models;
using Vision.MultiStream.Inference.Services.Snapshot;

namespace Vision.MultiStream.Inference.ViewModels
{
    public class SnapshotViewModel : BaseViewModel, IDisposable
    {
        private readonly ISnapshotDetector _detector;

        private string? _imagePath;
        private BitmapImage? _imageSource;
        private int _imageWidth;
        private int _imageHeight;
        private string _statusMessage = "이미지를 선택하세요.";
        private bool _isBusy;

        private CancellationTokenSource? _cts;

        public SnapshotViewModel(ISnapshotDetector detector)
        {
            _detector = detector;
            LoadImageCommand = new RelayCommand(LoadImage, CanLoadImage);
            DetectCommand = new AsyncRelayCommand(DetectAsync, CanDetect);
            CancelCommand = new RelayCommand(Cancel, CanCancel);
        }

        public ObservableCollection<Detection> Detections { get; } = new();

        public string? ImagePath
        {
            get => _imagePath;
            set
            {
                if (_imagePath == value)
                {
                    return;
                }
                _imagePath = value;
                OnPropertyChanged();
                DetectCommand.RaiseCanExecuteChanged();
            }
        }

        public BitmapImage? ImageSource
        {
            get => _imageSource;
            set
            {
                if (_imageSource == value)
                {
                    return;
                }
                _imageSource = value;
                OnPropertyChanged();
            }
        }

        public int ImageWidth
        {
            get => _imageWidth;
            set
            {
                if (_imageWidth == value)
                {
                    return;
                }
                _imageWidth = value;
                OnPropertyChanged();
            }
        }

        public int ImageHeight
        {
            get => _imageHeight;
            set
            {
                if (_imageHeight == value)
                {
                    return;
                }
                _imageHeight = value;
                OnPropertyChanged();
            }
        }

        public string StatusMessage
        {
            get => _statusMessage;
            set
            {
                if (_statusMessage == value)
                {
                    return;
                }
                _statusMessage = value;
                OnPropertyChanged();
            }
        }

        public bool IsBusy
        {
            get => _isBusy;
            set
            {
                if (_isBusy == value)
                {
                    return;
                }
                _isBusy = value;
                OnPropertyChanged();
                LoadImageCommand.RaiseCanExecuteChanged();
                DetectCommand.RaiseCanExecuteChanged();
                CancelCommand.RaiseCanExecuteChanged();
            }
        }

        public RelayCommand LoadImageCommand { get; }
        public AsyncRelayCommand DetectCommand { get; }
        public RelayCommand CancelCommand { get; }

        private bool CanLoadImage()
        {
            return !IsBusy;
        }

        private bool CanDetect()
        {
            return !IsBusy && !string.IsNullOrEmpty(ImagePath) && File.Exists(ImagePath);
        }

        private bool CanCancel()
        {
            return IsBusy;
        }

        private void LoadImage()
        {
            var dialog = new OpenFileDialog
            {
                Title = "이미지 선택",
                Filter = "이미지 파일 (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|모든 파일 (*.*)|*.*",
                CheckFileExists = true
            };

            if (dialog.ShowDialog() != true)
            {
                return;
            }

            try
            {
                // BitmapImage 를 파일 잠금 없이 로드 (CacheOption.OnLoad)
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.UriSource = new Uri(dialog.FileName);
                bmp.EndInit();
                bmp.Freeze();

                ImageSource = bmp;
                ImageWidth = bmp.PixelWidth;
                ImageHeight = bmp.PixelHeight;
                ImagePath = dialog.FileName;

                Detections.Clear();
                StatusMessage = $"로드됨: {Path.GetFileName(dialog.FileName)} ({bmp.PixelWidth}x{bmp.PixelHeight})";
            }
            catch (Exception ex)
            {
                StatusMessage = $"이미지 로드 실패: {ex.Message}";
            }
        }

        private void Cancel()
        {
            _cts?.Cancel();
            StatusMessage = "취소 요청됨...";
        }

        private async Task DetectAsync()
        {
            if (string.IsNullOrEmpty(ImagePath))
            {
                return;
            }

            _cts?.Dispose();
            _cts = new CancellationTokenSource();
            CancellationToken token = _cts.Token;

            IsBusy = true;
            Detections.Clear();
            StatusMessage = "추론 중...";

            var sw = Stopwatch.StartNew();
            try
            {
                IReadOnlyList<Detection> results = await _detector.DetectAsync(ImagePath, token);
                sw.Stop();

                foreach (Detection d in results)
                {
                    Detections.Add(d);
                }

                StatusMessage = $"완료: {results.Count}개 검출 ({sw.ElapsedMilliseconds} ms)";
            }
            catch (OperationCanceledException)
            {
                StatusMessage = "취소됨";
            }
            catch (Exception ex)
            {
                StatusMessage = $"오류: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }

        public void Dispose()
        {
            _cts?.Cancel();
            _cts?.Dispose();
            _cts = null;
        }
    }
}
