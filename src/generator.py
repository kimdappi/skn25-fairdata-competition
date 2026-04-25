"""
================================================================================
[공정위 AI 데이터 활용 공모전 - RAG 파이프라인 Step 4]
vLLM 기반 초고속 텍스트 생성 모듈 (Generator)
================================================================================

이 모듈은 RAG 시스템의 마지막 단계인 '답변 생성(Generation)' 역할을 전담합니다.
Retriever가 찾아준 Top 5 문서의 본문(Context)과 사용자의 질문(Query)을 조합하여
완벽한 프롬프트(Prompt)를 만들고, 8B 이하의 거대 언어 모델(LLM)에 전달하여 
최종 답변을 도출합니다.

[주요 역할 및 핵심 기술]
1. vLLM 엔진 적용 (30초 타임아웃 완벽 방어):
   - 순수 PyTorch/HuggingFace 파이프라인 대신 UC 버클리에서 개발한 'vLLM' 라이브러리를 사용합니다.
   - PagedAttention 기술을 통해 GPU 메모리 파편화를 막아 텍스트 생성 속도(Throughput)를 
     기존 대비 수 배 이상 비약적으로 끌어올려, 공모전의 '30초 제한'을 여유롭게 통과합니다.

2. A100 GPU 메모리 자원 분배 (OOM: Out of Memory 방지):
   - 동일한 GPU 내에 1차 검색용 임베딩 모델, 2차 검색용 ONNX Reranker, 그리고 거대한 
     LLM이 동시에 올라가야 합니다.
   - `gpu_memory_utilization` 파라미터를 조절하여(예: 0.5~0.6) LLM이 전체 VRAM을 
     독식하지 않도록 락(Lock)을 걸어줌으로써, 검색 모듈들이 작동할 메모리 공간을 양보합니다.
     이 방어 로직이 없으면 서버가 부팅되자마자 메모리 충돌로 다운됩니다.

3. 컨텍스트 길이 최적화 (`max_model_len`):
   - 5개의 공정위 의결서 청크를 프롬프트에 넣었을 때의 최대 길이를 계산하여 모델의 
     최대 토큰 길이를 제한합니다. 불필요한 메모리 할당을 줄여 속도를 높입니다.
================================================================================
"""

from vllm import LLM, SamplingParams

class RAGGenerator:
    def __init__(self, model_path: str, gpu_utilization: float = 0.55):
        """
        API 서버 부팅 시 1회 호출되어 vLLM 엔진을 GPU 메모리에 적재합니다.
        
        Args:
            model_path (str): 로컬에 다운로드된 LLM 가중치 폴더 경로 
                              (예: '../models/local_llm')
            gpu_utilization (float): LLM이 사용할 GPU VRAM의 최대 비율 (0.0 ~ 1.0)
                                     나머지 메모리는 Retriever(FAISS/Reranker)가 사용
        """
        print(f"Loading vLLM Engine from {model_path} (GPU Util: {gpu_utilization*100}%)...")
        
        # A100 GPU 환경에 맞춘 최적화 세팅
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",             # A100 텐서코어에 최적화된 데이터 타입 (빠르고 가벼움)
            max_model_len=4096,           # 의결서 5개 분량을 소화하면서도 메모리를 아끼는 적정 길이
            gpu_memory_utilization=gpu_utilization, # ⭐ RAG 통합 서버의 핵심 방어 로직
            enforce_eager=True,           # 메모리 부족 시 CUDA Graph 생성을 생략하여 안정성 확보
            trust_remote_code=True,       # EXAONE 등 최신 모델 로드를 위한 필수 옵션
            tensor_parallel_size=1        # A100 1장 사용 기준
        )
        
        # 답변 생성을 위한 샘플링 파라미터 (공모전 FAQ 권장 일관성 세팅)
        self.sampling_params = SamplingParams(
            temperature=0.0,              # 답변의 일관성을 위해 0 설정 (환각 최소화)
            max_tokens=512,               # 불필요하게 말을 길게 하여 30초를 초과하는 것을 방지
            top_p=0.95
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """
        LLM에게 전달할 최종 프롬프트를 조립합니다.
        (사용하는 모델-EXAONE, Llama3 등-의 Chat Template 규격에 맞춰 수정 가능합니다)
        """
        system_instruction = (
            "당신은 공정거래위원회 의결서 데이터를 기반으로 기업 컴플라이언스 및 법무 검토를 "
            "지원하는 전문 AI 어시스턴트입니다. 제공된 [문서] 내용에만 근거하여 [질문]에 답변하세요. "
            "문서에 없는 내용은 지어내지 마세요."
        )
        
        prompt = (
            f"{system_instruction}\n\n"
            f"[문서]\n{context}\n\n"
            f"[질문]\n{query}\n\n"
            f"답변:"
        )
        return prompt

    def generate_answer(self, query: str, context: str) -> str:
        """
        조립된 프롬프트를 vLLM 엔진에 전달하여 고속으로 텍스트를 생성합니다.
        
        Args:
            query (str): 사용자의 원본 질문
            context (str): Retriever가 찾아준 Top 5 청크의 병합된 텍스트
            
        Returns:
            str: LLM이 생성한 최종 답변 텍스트
        """
        prompt = self._build_prompt(query, context)
        
        # vLLM을 통한 초고속 텍스트 생성 (use_tqdm=False로 불필요한 로그 생략)
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        
        # 첫 번째 배치(배치 크기 1)의 첫 번째 생성 결과 텍스트 추출
        generated_text = outputs[0].outputs[0].text.strip()
        
        return generated_text


if __name__ == "__main__":
    # 단독 테스트 로직
    # 주의: vLLM은 실행 시 GPU 메모리를 즉시 점유합니다.
    import os
    
    LLM_MODEL_PATH = '../models/local_llm'
    
    if os.path.exists(LLM_MODEL_PATH):
        print("\n--- Generator(vLLM) 모듈 초기화 테스트 ---")
        try:
            generator = RAGGenerator(LLM_MODEL_PATH, gpu_utilization=0.4) # 테스트 시엔 메모리 점유율을 낮춤
            
            test_query = "부당한 공동행위를 한 기업의 위반 유형은?"
            test_context = (
                "[문서제목: ㈜팬택의 부당한 공동행위 건, 피심인: ㈜팬택, 위반유형: 공동행위]\n"
                "1. 피심인은 경쟁사와 가격을 담합하여 부당한 이익을 취득하였다."
            )
            
            print(f"\n[테스트 질의]: {test_query}")
            print("답변 생성 중...")
            
            answer = generator.generate_answer(test_query, test_context)
            print(f"\n[생성된 답변]:\n{answer}")
            
        except Exception as e:
            print(f"테스트 실패: {e}")
    else:
        print(f"테스트 건너뜀: {LLM_MODEL_PATH} 경로에 다운로드된 LLM 모델이 없습니다.")