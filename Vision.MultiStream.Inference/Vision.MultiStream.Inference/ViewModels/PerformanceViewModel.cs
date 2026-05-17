using System;
using System.Diagnostics;
using System.Threading;
using System.Windows.Threading;
using Vision.MultiStream.Inference.Common;

namespace Vision.MultiStream.Inference.ViewModels
{
    /// <summary>
    /// 현재 프로세스의 CPU / 메모리 / GC 수집 횟수 / GC 일시정지 시간을 주기적으로 갱신.
    ///
    /// GC 일시정지 측정 원리 (센티넬 스레드):
    ///   .NET GC는 Stop-the-World 방식으로 모든 managed 스레드를 동시에 멈춘다.
    ///   5ms마다 자는 백그라운드 스레드도 GC 동안 같이 멈추므로,
    ///   실제 잠든 시간 − 예상 5ms = GC 일시정지 시간 으로 추정할 수 있다.
    /// </summary>
    public sealed class PerformanceViewModel : BaseViewModel, IDisposable
    {
        private readonly Process _proc = Process.GetCurrentProcess();
        private readonly DispatcherTimer _timer;
        private readonly Thread _sentinelThread;

        private TimeSpan _lastCpuTime;
        private DateTime _lastSampleTime;
        private volatile bool _disposed;

        // 센티넬 스레드가 쓰고, UI 타이머가 읽음 (표시용이라 lock 불필요)
        private long _lastGcPauseMs;
        private long _maxGcPauseMs;
        private long _pauseCount;
        private long _totalPauseMs;

        private double _cpuPercent;
        private long _memoryMb;
        private int _gen0;
        private int _gen1;
        private int _gen2;
        private long _lastGcPauseMsDisplay;
        private long _maxGcPauseMsDisplay;
        private long _pauseCountDisplay;
        private long _totalPauseMsDisplay;

        public double CpuPercent
        {
            get => _cpuPercent;
            private set { _cpuPercent = value; OnPropertyChanged(); }
        }

        public long MemoryMb
        {
            get => _memoryMb;
            private set { _memoryMb = value; OnPropertyChanged(); }
        }

        public int Gen0
        {
            get => _gen0;
            private set { _gen0 = value; OnPropertyChanged(); }
        }

        public int Gen1
        {
            get => _gen1;
            private set { _gen1 = value; OnPropertyChanged(); }
        }

        public int Gen2
        {
            get => _gen2;
            private set { _gen2 = value; OnPropertyChanged(); }
        }

        public long LastGcPauseMs
        {
            get => _lastGcPauseMsDisplay;
            private set { _lastGcPauseMsDisplay = value; OnPropertyChanged(); }
        }

        public long MaxGcPauseMs
        {
            get => _maxGcPauseMsDisplay;
            private set { _maxGcPauseMsDisplay = value; OnPropertyChanged(); }
        }

        public long PauseCount
        {
            get => _pauseCountDisplay;
            private set { _pauseCountDisplay = value; OnPropertyChanged(); }
        }

        public long TotalPauseMs
        {
            get => _totalPauseMsDisplay;
            private set { _totalPauseMsDisplay = value; OnPropertyChanged(); }
        }

        public PerformanceViewModel()
        {
            _proc.Refresh();
            _lastCpuTime = _proc.TotalProcessorTime;
            _lastSampleTime = DateTime.UtcNow;

            _sentinelThread = new Thread(SentinelLoop) { IsBackground = true, Name = "GcSentinel" };
            _sentinelThread.Start();

            _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
            _timer.Tick += OnTick;
            _timer.Start();
        }

        // 5ms마다 실제 경과 시간을 측정해 GC 일시정지를 감지하는 센티넬 루프.
        private void SentinelLoop()
        {
            const int sleepMs = 5;
            const int thresholdMs = 10; // 10ms 초과 = GC 일시정지로 판정

            var sw = Stopwatch.StartNew();
            long prev = sw.ElapsedMilliseconds;

            while (!_disposed)
            {
                Thread.Sleep(sleepMs);

                long now = sw.ElapsedMilliseconds;
                long delta = now - prev;
                prev = now;

                long excess = delta - sleepMs;
                if (excess >= thresholdMs)
                {
                    Interlocked.Exchange(ref _lastGcPauseMs, excess);
                    Interlocked.Increment(ref _pauseCount);
                    Interlocked.Add(ref _totalPauseMs, excess);

                    long current = Interlocked.Read(ref _maxGcPauseMs);
                    if (excess > current)
                    {
                        Interlocked.CompareExchange(ref _maxGcPauseMs, excess, current);
                    }
                }
            }
        }

        private void OnTick(object? sender, EventArgs e)
        {
            _proc.Refresh();

            var now = DateTime.UtcNow;
            var cpuNow = _proc.TotalProcessorTime;

            double wallMs = (now - _lastSampleTime).TotalMilliseconds;
            double cpuMs = (cpuNow - _lastCpuTime).TotalMilliseconds;
            CpuPercent = wallMs > 0
                ? Math.Min(cpuMs / (wallMs * Environment.ProcessorCount) * 100.0, 100.0)
                : 0;

            _lastSampleTime = now;
            _lastCpuTime = cpuNow;

            MemoryMb = _proc.WorkingSet64 / 1024 / 1024;

            Gen0 = GC.CollectionCount(0);
            Gen1 = GC.CollectionCount(1);
            Gen2 = GC.CollectionCount(2);

            LastGcPauseMs = Interlocked.Read(ref _lastGcPauseMs);
            MaxGcPauseMs = Interlocked.Read(ref _maxGcPauseMs);
            PauseCount = Interlocked.Read(ref _pauseCount);
            TotalPauseMs = Interlocked.Read(ref _totalPauseMs);
        }

        public void Dispose()
        {
            _disposed = true;
            _timer.Stop();
            _proc.Dispose();
        }
    }
}
