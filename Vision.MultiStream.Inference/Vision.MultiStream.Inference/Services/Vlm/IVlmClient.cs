using System.Threading;
using System.Threading.Tasks;

namespace Vision.MultiStream.Inference.Services.Vlm
{
    /// <summary>
    /// 비전-언어 모델(VLM) 호출 경계. "JPEG + 프롬프트 → 묘사 텍스트".
    /// 백엔드(Ollama LLaVA → llama.cpp → 직접 ONNX)를 갈아끼울 수 있게 이 한 군데만 인터페이스로 둔다.
    /// </summary>
    public interface IVlmClient
    {
        Task<string> DescribeAsync(byte[] jpeg, string prompt, CancellationToken cancellationToken = default);

        /// <summary>
        /// [Phase 3.5] 모델을 백엔드 메모리에 미리 적재(워밍업). 첫 실제 호출의 콜드 로딩 지연을
        /// 앱 시작 시점으로 당겨 사용자가 체감하지 않게 한다. 토큰 생성 없이 가중치 로드만 수행.
        /// </summary>
        Task WarmUpAsync(CancellationToken cancellationToken = default);
    }
}
