using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;

namespace Vision.MultiStream.Inference.Common
{
    public static class PerfProbe
    {
        private sealed class Counter
        {
            public long Count;
            public long TotalTicks;
            public long MaxTicks;
            public long LastReportTicks;
        }

        private readonly struct Scope : IDisposable
        {
            private readonly string _name;
            private readonly long _startTicks;

            public Scope(string name)
            {
                _name = name;
                _startTicks = Stopwatch.GetTimestamp();
            }

            public void Dispose()
            {
                Record(_name, Stopwatch.GetTimestamp() - _startTicks);
            }
        }

        private static readonly ConcurrentDictionary<string, Counter> Counters = new();
        private static readonly double TickToMs = 1000.0 / Stopwatch.Frequency;
        private static int _enabled = 1;

        public static bool Enabled
        {
            get => Volatile.Read(ref _enabled) != 0;
            set => Volatile.Write(ref _enabled, value ? 1 : 0);
        }

        public static IDisposable Measure(string name)
        {
            return Enabled ? new Scope(name) : EmptyDisposable.Instance;
        }

        // 이미 외부에서 잰 경과 시간(ms)을 집계에 넣는다. using 스코프로 감쌀 수 없는 경우
        // (예: 네이티브 엔진이 ms 값만 돌려줄 때) 사용. 집계/1초 롤업 로직은 Measure 와 공유.
        public static void RecordMs(string name, double elapsedMs)
        {
            if (!Enabled)
            {
                return;
            }

            long elapsedTicks = (long)(elapsedMs / TickToMs);
            Record(name, elapsedTicks);
        }

        private static void Record(string name, long elapsedTicks)
        {
            Counter counter = Counters.GetOrAdd(name, _ =>
            {
                return new Counter
                {
                    LastReportTicks = Stopwatch.GetTimestamp()
                };
            });

            Interlocked.Increment(ref counter.Count);
            Interlocked.Add(ref counter.TotalTicks, elapsedTicks);
            UpdateMax(counter, elapsedTicks);

            long now = Stopwatch.GetTimestamp();
            long last = Volatile.Read(ref counter.LastReportTicks);
            if ((now - last) * TickToMs < 1000)
            {
                return;
            }

            if (Interlocked.CompareExchange(ref counter.LastReportTicks, now, last) != last)
            {
                return;
            }

            long count = Interlocked.Exchange(ref counter.Count, 0);
            long total = Interlocked.Exchange(ref counter.TotalTicks, 0);
            long max = Interlocked.Exchange(ref counter.MaxTicks, 0);
            if (count <= 0)
            {
                return;
            }

            double avgMs = total * TickToMs / count;
            double maxMs = max * TickToMs;
            Debug.WriteLine(
                $"[PerfProbe] {name}: count={count}, avg={avgMs:F3} ms, max={maxMs:F3} ms");
        }

        private static void UpdateMax(Counter counter, long elapsedTicks)
        {
            while (true)
            {
                long current = Volatile.Read(ref counter.MaxTicks);
                if (elapsedTicks <= current)
                {
                    return;
                }

                if (Interlocked.CompareExchange(ref counter.MaxTicks, elapsedTicks, current) == current)
                {
                    return;
                }
            }
        }

        private sealed class EmptyDisposable : IDisposable
        {
            public static readonly EmptyDisposable Instance = new();

            public void Dispose()
            {
            }
        }
    }
}
