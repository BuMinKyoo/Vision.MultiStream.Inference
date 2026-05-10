using System;
using System.Diagnostics;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// A/V 동기화용 마스터 클럭.
    ///
    /// 첫 프레임 도착 시점을 wall clock 에 anchor 한 뒤,
    /// 이후 모든 프레임은 (pts - firstPts) 가 (now - startTime) 과 같아질 때 출력되도록 게이트.
    ///
    /// 사용 패턴:
    ///   if (!clock.IsAnchored) { clock.Anchor(pts); }
    ///   TimeSpan delay = clock.GetDelay(pts);
    ///     delay > 0  → 미래(아직 일찍 옴, sleep 필요)
    ///     delay < 0  → 과거(이미 늦음, drop 후보)
    /// </summary>
    public sealed class MediaClock
    {
        private readonly object _gate = new();
        private bool _anchored;
        private double _firstPtsSeconds;
        private long _startTicks;

        public bool IsAnchored
        {
            get
            {
                lock (_gate)
                {
                    return _anchored;
                }
            }
        }

        public double FirstPtsSeconds
        {
            get
            {
                lock (_gate)
                {
                    return _firstPtsSeconds;
                }
            }
        }

        public void Anchor(double firstPtsSeconds)
        {
            lock (_gate)
            {
                if (_anchored)
                {
                    return;
                }
                _firstPtsSeconds = firstPtsSeconds;
                _startTicks = Stopwatch.GetTimestamp();
                _anchored = true;
            }
        }

        public void Reset()
        {
            lock (_gate)
            {
                _anchored = false;
                _firstPtsSeconds = 0;
                _startTicks = 0;
            }
        }

        public TimeSpan GetDelay(double ptsSeconds)
        {
            lock (_gate)
            {
                if (!_anchored)
                {
                    return TimeSpan.Zero;
                }

                // anchor PTS = 1.0, 들어온 프레임 PTS = 1.5 / targetSec = 0.5초 (앵커 0.5초 후가 이 프레임 시점)
                double targetSec = ptsSeconds - _firstPtsSeconds;

                // 월 클럭 타임라인에서 "지금 실제로 얼마나 흘렀는지"
                double actualSec = (Stopwatch.GetTimestamp() - _startTicks) / (double)Stopwatch.Frequency;
                return TimeSpan.FromSeconds(targetSec - actualSec);

                /*
                 
                양수 (X > Y) │ 프레임이 일찍 도착함. 아직 출력 시점 아님 │ sleep(차이)로 대기
                0 근처       │ 거의 정확한 시점                          │ 즉시 출력            
                음수 (X < Y) │ 출력 시점 이미 지나감. 늦었음             │ 너무 많이 늦으면 drop

                 */
            }
        }
    }
}
