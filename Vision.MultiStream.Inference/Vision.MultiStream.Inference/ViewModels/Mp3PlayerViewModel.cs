using System;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Threading;
using Microsoft.Win32;
using NAudio.Wave;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Models;

namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 로컬 MP3/오디오 파일을 재생하는 플레이어 ViewModel.
    /// NAudio 의 AudioFileReader + WaveOutEvent 를 직접 사용한다 (RTSP/FFmpeg 경로와 무관).
    /// 곡이 끝까지 재생되면 플레이리스트의 다음 곡으로 자동 진행한다.
    /// </summary>
    public sealed class Mp3PlayerViewModel : BaseViewModel, IDisposable
    {
        private readonly Dispatcher _dispatcher;
        private readonly DispatcherTimer _positionTimer;

        private WaveOutEvent? _waveOut;
        private AudioFileReader? _reader;

        private Mp3Track? _selectedTrack;
        private Mp3Track? _currentTrack;
        private bool _isPlaying;
        private bool _isPaused;
        private double _positionSeconds;
        private double _durationSeconds;
        private double _volume = 0.5;
        private string _statusMessage = "대기";

        // 타이머가 갱신한 Position 변경인지(=seek 하면 안 됨) 사용자가 슬라이더를 움직인 것인지 구분.
        private bool _updatingPositionFromTimer;
        // PlaybackStopped 가 "곡 끝까지 재생" 인지 "사용자 정지/곡 전환" 인지 구분.
        private bool _advanceOnStop;

        public Mp3PlayerViewModel()
        {
            _dispatcher = Application.Current.Dispatcher;
            _positionTimer = new DispatcherTimer(DispatcherPriority.Background, _dispatcher)
            {
                Interval = TimeSpan.FromMilliseconds(250)
            };
            _positionTimer.Tick += OnPositionTick;

            AddFilesCommand = new RelayCommand(AddFiles);
            RemoveSelectedCommand = new RelayCommand(RemoveSelected, () => _selectedTrack != null);
            ClearPlaylistCommand = new RelayCommand(ClearPlaylist, () => Playlist.Count > 0);
            PlayCommand = new RelayCommand(Play, CanPlay);
            PauseCommand = new RelayCommand(Pause, () => _isPlaying);
            StopCommand = new RelayCommand(Stop, () => _isPlaying || _isPaused);
            NextCommand = new RelayCommand(Next, () => Playlist.Count > 0);
            PreviousCommand = new RelayCommand(Previous, () => Playlist.Count > 0);
        }

        public ObservableCollection<Mp3Track> Playlist { get; } = new();

        public RelayCommand AddFilesCommand { get; }
        public RelayCommand RemoveSelectedCommand { get; }
        public RelayCommand ClearPlaylistCommand { get; }
        public RelayCommand PlayCommand { get; }
        public RelayCommand PauseCommand { get; }
        public RelayCommand StopCommand { get; }
        public RelayCommand NextCommand { get; }
        public RelayCommand PreviousCommand { get; }

        public Mp3Track? SelectedTrack
        {
            get => _selectedTrack;
            set
            {
                if (_selectedTrack == value)
                {
                    return;
                }
                _selectedTrack = value;
                OnPropertyChanged();
                RemoveSelectedCommand.RaiseCanExecuteChanged();
                PlayCommand.RaiseCanExecuteChanged();
            }
        }

        public bool IsPlaying
        {
            get => _isPlaying;
            private set
            {
                if (_isPlaying == value)
                {
                    return;
                }
                _isPlaying = value;
                OnPropertyChanged();
            }
        }

        public bool IsPaused
        {
            get => _isPaused;
            private set
            {
                if (_isPaused == value)
                {
                    return;
                }
                _isPaused = value;
                OnPropertyChanged();
            }
        }

        public double PositionSeconds
        {
            get => _positionSeconds;
            set
            {
                if (Math.Abs(_positionSeconds - value) < 0.05)
                {
                    return;
                }
                _positionSeconds = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(PositionText));

                // 타이머가 아니라 사용자가 슬라이더를 움직인 경우에만 seek.
                if (!_updatingPositionFromTimer && _reader != null)
                {
                    _reader.CurrentTime = TimeSpan.FromSeconds(value);
                }
            }
        }

        public double DurationSeconds
        {
            get => _durationSeconds;
            private set
            {
                if (Math.Abs(_durationSeconds - value) < 0.05)
                {
                    return;
                }
                _durationSeconds = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(DurationText));
            }
        }

        public string PositionText => FormatTime(_positionSeconds);
        public string DurationText => FormatTime(_durationSeconds);

        public double Volume
        {
            get => _volume;
            set
            {
                double clamped = Math.Clamp(value, 0.0, 1.0);
                if (Math.Abs(_volume - clamped) < 0.001)
                {
                    return;
                }
                _volume = clamped;
                OnPropertyChanged();
                if (_reader != null)
                {
                    _reader.Volume = (float)clamped;
                }
            }
        }

        public string StatusMessage
        {
            get => _statusMessage;
            private set
            {
                if (_statusMessage == value)
                {
                    return;
                }
                _statusMessage = value;
                OnPropertyChanged();
            }
        }

        private void AddFiles()
        {
            var dialog = new OpenFileDialog
            {
                Title = "오디오 파일 선택",
                Filter = "오디오 파일|*.mp3;*.wav;*.m4a;*.aac;*.flac;*.wma|모든 파일|*.*",
                Multiselect = true
            };
            if (dialog.ShowDialog() != true)
            {
                return;
            }

            foreach (string path in dialog.FileNames)
            {
                Playlist.Add(new Mp3Track(path));
            }

            if (_selectedTrack == null && Playlist.Count > 0)
            {
                SelectedTrack = Playlist[0];
            }
            RaisePlaylistStates();
        }

        private void RemoveSelected()
        {
            Mp3Track? track = _selectedTrack;
            if (track == null)
            {
                return;
            }

            int idx = Playlist.IndexOf(track);
            // 재생 중인 곡을 지우면 정지.
            if (track == _currentTrack)
            {
                Stop();
                _currentTrack = null;
            }
            Playlist.Remove(track);

            if (Playlist.Count > 0)
            {
                SelectedTrack = Playlist[Math.Min(idx, Playlist.Count - 1)];
            }
            else
            {
                SelectedTrack = null;
            }
            RaisePlaylistStates();
        }

        private void ClearPlaylist()
        {
            Stop();
            _currentTrack = null;
            Playlist.Clear();
            SelectedTrack = null;
            RaisePlaylistStates();
        }

        private bool CanPlay()
        {
            if (_isPaused)
            {
                return true;
            }
            return !_isPlaying && Playlist.Count > 0;
        }

        private void Play()
        {
            // 일시정지 상태면 이어서 재생.
            if (_isPaused && _waveOut != null)
            {
                _waveOut.Play();
                IsPaused = false;
                IsPlaying = true;
                _positionTimer.Start();
                StatusMessage = $"재생 중: {_currentTrack?.Name}";
                RaiseTransportStates();
                return;
            }

            Mp3Track? track = _selectedTrack ?? (Playlist.Count > 0 ? Playlist[0] : null);
            if (track == null)
            {
                return;
            }
            PlayTrack(track);
        }

        private void PlayTrack(Mp3Track track)
        {
            DisposePlayback();
            try
            {
                _reader = new AudioFileReader(track.FilePath)
                {
                    Volume = (float)_volume
                };
                _waveOut = new WaveOutEvent();
                _waveOut.PlaybackStopped += OnPlaybackStopped;
                _waveOut.Init(_reader);
                _waveOut.Play();
                _advanceOnStop = true;

                _currentTrack = track;
                SelectedTrack = track;
                DurationSeconds = _reader.TotalTime.TotalSeconds;
                _updatingPositionFromTimer = true;
                PositionSeconds = 0;
                _updatingPositionFromTimer = false;
                IsPlaying = true;
                IsPaused = false;
                _positionTimer.Start();
                StatusMessage = $"재생 중: {track.Name}";
            }
            catch (Exception ex)
            {
                DisposePlayback();
                IsPlaying = false;
                IsPaused = false;
                StatusMessage = $"재생 실패: {ex.Message}";
            }
            RaiseTransportStates();
        }

        private void Pause()
        {
            if (_waveOut == null || !_isPlaying)
            {
                return;
            }
            _waveOut.Pause();
            _positionTimer.Stop();
            IsPlaying = false;
            IsPaused = true;
            StatusMessage = $"일시정지: {_currentTrack?.Name}";
            RaiseTransportStates();
        }

        private void Stop()
        {
            DisposePlayback();
            IsPlaying = false;
            IsPaused = false;
            _updatingPositionFromTimer = true;
            PositionSeconds = 0;
            _updatingPositionFromTimer = false;
            if (_currentTrack != null)
            {
                StatusMessage = "정지";
            }
            RaiseTransportStates();
        }

        private void Next()
        {
            Mp3Track? reference = _currentTrack ?? _selectedTrack;
            int idx = reference != null ? Playlist.IndexOf(reference) : -1;
            int nextIdx = idx + 1;
            if (nextIdx >= Playlist.Count)
            {
                // 마지막 곡 다음 → 정지 (반복 없음).
                Stop();
                return;
            }
            PlayTrack(Playlist[nextIdx]);
        }

        private void Previous()
        {
            Mp3Track? reference = _currentTrack ?? _selectedTrack;
            int idx = reference != null ? Playlist.IndexOf(reference) : -1;
            int prevIdx = idx - 1;
            if (prevIdx < 0)
            {
                // 첫 곡 이전 → 현재 곡을 처음부터 다시.
                if (reference != null)
                {
                    PlayTrack(reference);
                }
                return;
            }
            PlayTrack(Playlist[prevIdx]);
        }

        private void OnPlaybackStopped(object? sender, StoppedEventArgs e)
        {
            // 사용자 정지/곡 전환은 DisposePlayback 에서 _advanceOnStop=false 로 만든 뒤 일어난다.
            // 여기서 _advanceOnStop 이 true 면 "곡이 끝까지 재생됐다" 는 의미.
            _dispatcher.BeginInvoke(() =>
            {
                if (!_advanceOnStop)
                {
                    return;
                }
                _advanceOnStop = false;

                if (e.Exception != null)
                {
                    StatusMessage = $"재생 오류: {e.Exception.Message}";
                    Stop();
                    return;
                }
                Next();
            });
        }

        private void OnPositionTick(object? sender, EventArgs e)
        {
            if (_reader == null || !_isPlaying)
            {
                return;
            }
            _updatingPositionFromTimer = true;
            PositionSeconds = _reader.CurrentTime.TotalSeconds;
            _updatingPositionFromTimer = false;
        }

        private void DisposePlayback()
        {
            _advanceOnStop = false;
            _positionTimer.Stop();
            if (_waveOut != null)
            {
                _waveOut.PlaybackStopped -= OnPlaybackStopped;
                try
                {
                    _waveOut.Stop();
                }
                catch
                {
                    // 디바이스가 이미 사라진 경우 등 — 무시.
                }
                _waveOut.Dispose();
                _waveOut = null;
            }
            _reader?.Dispose();
            _reader = null;
        }

        private void RaiseTransportStates()
        {
            PlayCommand.RaiseCanExecuteChanged();
            PauseCommand.RaiseCanExecuteChanged();
            StopCommand.RaiseCanExecuteChanged();
            NextCommand.RaiseCanExecuteChanged();
            PreviousCommand.RaiseCanExecuteChanged();
        }

        private void RaisePlaylistStates()
        {
            ClearPlaylistCommand.RaiseCanExecuteChanged();
            RemoveSelectedCommand.RaiseCanExecuteChanged();
            PlayCommand.RaiseCanExecuteChanged();
            NextCommand.RaiseCanExecuteChanged();
            PreviousCommand.RaiseCanExecuteChanged();
        }

        private static string FormatTime(double seconds)
        {
            if (double.IsNaN(seconds) || seconds < 0)
            {
                seconds = 0;
            }
            TimeSpan ts = TimeSpan.FromSeconds(seconds);
            return $"{(int)ts.TotalMinutes:D2}:{ts.Seconds:D2}";
        }

        public void Dispose()
        {
            _positionTimer.Tick -= OnPositionTick;
            DisposePlayback();
        }
    }
}
