using System;
using System.Diagnostics;
using Vision.MultiStream.Inference.Services.Audio;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// A/V 동기화용 마스터 클럭. 두 가지 모드를 지원한다.
    ///
    /// 1) Audio-master 모드 (영상+오디오):
    ///    사운드카드가 페이싱하는 실제 오디오 재생 위치를 마스터로 삼는다.
    ///    마스터 PTS = (마지막 push 된 오디오 PTS) − (출력 버퍼에 쌓인 시간).
    ///    오디오는 게이팅 없이 흘려보내고, 비디오가 그 시각에 맞춰 표시된다.
    ///
    /// 2) Wall-clock 모드 (영상만):
    ///    비디오 첫 프레임 PTS 를 wall clock 에 anchor 한다.
    ///    마스터 PTS = firstPts + wallElapsed.
    ///
    /// 비디오 디코더 사용 패턴 (양 모드 공통):
    ///   if (!clock.IsReady) { clock.Anchor(pts); } // audio 모드면 내부적으로 무시됨
    ///   TimeSpan delay = clock.GetDelay(pts);
    ///     delay > 0 → 미래(일찍 옴, sleep)
    ///     delay < 0 → 과거(늦음, drop 후보)
    /// </summary>
    public sealed class MediaClock
    {
        private readonly object _gate = new();

        // 모드 결정
        private bool _audioMaster = false;
        private IAudioOutput? _audioOutput;
        private double _lastAudioPushedPts = double.NaN;

        // Wall-clock 모드 상태
        private bool _wallAnchored;
        private double _wallFirstPts;
        private long _wallStartTicks;

        public bool IsAudioMaster
        {
            get
            {
                lock (_gate)
                {
                    return _audioMaster;
                }
            }
        }

        /// <summary>
        /// 마스터가 시각을 답할 준비가 됐는지.
        /// Audio 모드: 오디오가 한 번이라도 push 됐는지.
        /// Wall-clock 모드: anchor 됐는지.
        /// </summary>
        public bool IsReady
        {
            get
            {
                lock (_gate)
                {
                    if (_audioMaster)
                    {
                        return !double.IsNaN(_lastAudioPushedPts);
                    }
                    return _wallAnchored;
                }
            }
        }

        /// <summary>
        /// 오디오 마스터 모드로 전환. Start 시점에 오디오가 활성이면 호출한다.
        /// </summary>
        public void UseAudioMaster(IAudioOutput audioOutput)
        {
            lock (_gate)
            {
                _audioMaster = true;
                _audioOutput = audioOutput;
            }
        }

        /// <summary>
        /// 비디오 첫 프레임에서 호출. Audio-master 모드면 무시된다.
        /// </summary>
        public void Anchor(double firstPtsSeconds)
        {
            lock (_gate)
            {
                if (_audioMaster)
                {
                    return;
                }
                if (_wallAnchored)
                {
                    return;
                }

                // 비디오의 첫 pts 기준으로 세팅
                _wallFirstPts = firstPtsSeconds;
                _wallStartTicks = Stopwatch.GetTimestamp();
                _wallAnchored = true;
            }
        }

        /// <summary>
        /// 오디오 디코더가 output.Push 직후 호출. 마지막 push 된 PTS 를 갱신한다.
        /// </summary>
        public void OnAudioPushed(double ptsSeconds)
        {
            lock (_gate)
            {
                _lastAudioPushedPts = ptsSeconds;
            }
        }

        public void Reset()
        {
            lock (_gate)
            {
                _audioMaster = false;
                _audioOutput = null;
                _lastAudioPushedPts = double.NaN;
                _wallAnchored = false;
                _wallFirstPts = 0;
                _wallStartTicks = 0;
            }
        }

        /// <summary>
        /// 현재 마스터 PTS(초). 준비 안됐으면 NaN.
        /// </summary>
        public double GetMasterPtsSeconds()
        {
            lock (_gate)
            {
                if (_audioMaster)
                {
                    if (double.IsNaN(_lastAudioPushedPts))
                    {
                        return double.NaN;
                    }
                    int bufferedMs = _audioOutput?.BufferedMs ?? 0;
                    // 마지막 push PTS 에서 아직 스피커로 못 나간 양만큼 빼면 "지금 들리는 PTS".
                    return _lastAudioPushedPts - bufferedMs / 1000.0;
                }

                if (_wallAnchored)
                {
                    // 예를 들어 첫 비디오 프레임 PTS가 100.0초(스트림 시작부터 100초 지점)이고, 재생 시작 후 실제로 2.5초가 흘렀다면: masterPts = 100.0 + 2.5 = 102.5초, "지금 이 순간 재생되어야 할 스트림 위치"
                    double wallElapsed = (Stopwatch.GetTimestamp() - _wallStartTicks) / (double)Stopwatch.Frequency;
                    return _wallFirstPts + wallElapsed;
                }
                else
                {
                    return double.NaN;
                }
                
            }
        }

        /// <summary>
        /// 비디오 PTS 기준 출력 시점까지 남은 시간.
        /// 마스터 준비 안됐으면 Zero (즉시 출력).
        /// </summary>
        public TimeSpan GetDelay(double ptsSeconds)
        {
            double master = GetMasterPtsSeconds();
            if (double.IsNaN(master))
            {
                return TimeSpan.Zero;
            }
            return TimeSpan.FromSeconds(ptsSeconds - master);
        }
    }
}
