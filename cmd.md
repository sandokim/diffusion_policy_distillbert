export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

conda_environment.yaml -> cuda 12.8로 변경

conda env create -f conda_environment.yaml
pip install 'huggingface_hub<0.24.0'
pip install --upgrade wandb


(A) 정책 클래스 내부: visual feature 생성 직후에 FiLM 적용
대상 파일(레포 내 경로):
diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py

여기서 보통 흐름이:
1. observation에서 이미지/상태를 꺼냄
2. image encoder(visual backbone)로 feature 뽑음
3. obs_steps만큼 쌓아서 global_cond 만들고
4. conditional UNet에 global_cond 전달
→ 2와 3 사이에 FiLM을 끼우면 “language를 visual backbone feature에 integrate”가 됩니다.


# TinyVLA
Secondly, the vanilla DP does not incorporate language instructions. Therefore, following RT-1 [23] and YAY [39], we integrate language information into the visual backbone using FiLM [40].

[40] Film: Visual reasoning with a general conditioning layer


# CoT-VLA
Diffusion Policy에 DistillBERT language embeddings을 넣어서 condition을 줬음
Diffusion Policy: The implementation incorporates action chunking and proprioception while conditioning on DistillBERT [52] language embeddings.
[52] Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter.

## 구현 요약 (TinyVLA / CoT-VLA 스타일 DP)

- **새로 추가한 파일 경로**
  - `diffusion_policy/model/language/film.py`  
    - `FiMLayer(feature_dim, cond_dim, hidden_dim)`  
    - 입력: 시각 feature `x (B, To, D_feat)`, 언어 임베딩 `cond (B, D_lang)` 또는 `(B, To, D_lang)`  
    - 출력: `x`에 대해 `FiLM(x, cond) = x * (1 + γ) + β` 적용된 feature
  - `diffusion_policy/model/language/distilbert_encoder.py`  
    - `DistilBertLanguageEncoder`  
    - HuggingFace `distilbert-base-uncased`로부터 CLS 임베딩을 뽑아서 `(B, 768)` 텐서 반환

- **수정한 주요 코드**
  - `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`
    - `__init__`에 언어 옵션 추가
      - `use_language: bool = False`, `language_dim: int = 512`, `film_hidden_dim: int = 512`
      - `obs_encoder` 출력 차원(`obs_feature_dim`)과 `language_dim`으로 `FiMLayer` 초기화
    - `predict_action(self, obs_dict, language_emb=None)`  
      - 기존: `obs_encoder` → `nobs_features (B*To, Do)` → `global_cond = (B, To*Do)`  
      - 변경: `nobs_features`를 `(B, To, Do)`로 reshape 후  
        - `use_language=True` & `language_emb`가 있을 때 `FiMLayer`로 modulation  
        - 다시 `(B, To*Do)`로 펼쳐 `global_cond`에 사용  
      - `language_emb` shape: `(B, language_dim)` 또는 `(B, To, language_dim)`
    - `compute_loss(self, batch)`  
      - `language_emb = batch.get('language_emb', None)`  
      - `obs_as_global_cond=True`일 때, 학습 시에도 위와 같은 방식으로 FiLM 적용 후 `global_cond` 생성
  - `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`
    - GPU 디바이스에서 DistilBERT 인코더 준비:
      - `language.use_distilbert=True`이고 `training.language_instruction`이 있을 때  
        `DistilBertLanguageEncoder(model_name, device)` 생성
    - 학습 루프에서:
      - 각 batch 마다 배치 크기 `B`를 기준으로  
        `texts = [language_instruction] * B`  
        `language_emb = encoder.encode(texts)` → `batch['language_emb']`로 추가
    - validation, sampling에서도 동일하게 `language_emb`를 생성해서  
      - `self.model.compute_loss(batch)` / `policy.predict_action(obs_dict, language_emb)`에 전달
  - `diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml`
    - `policy` 블록:
      - `use_language: True`, `language_dim: 768`, `film_hidden_dim: 512`
    - `training` 블록:
      - `language_instruction: "pick up the object"` (예시용 공통 instruction)
    - `language` 블록:
      - `use_distilbert: True`, `model_name: "distilbert-base-uncased"`
  - `conda_environment.yaml`
    - pip 패키지에 `transformers==4.36.0` 추가 (DistilBERT 사용)

- **핵심 아이디어**
  - **CoT-VLA 스타일**: DistilBERT CLS 임베딩을 `language_emb`로 사용 (`D_lang=768`)  
  - **TinyVLA / RT-1 스타일**: `obs_encoder`가 만든 **시각 feature (B, To, Do)** 에  
    `FiMLayer`로 언어 정보를 multiplicative/additive modulation 하여  
    언어가 통합된 visual global feature를 `ConditionalUnet1D`의 `global_cond`로 전달  
  - 학습 시: `train_diffusion_unet_hybrid_workspace`가 DistilBERT를 자동 호출해  
    batch마다 `batch['language_emb']`를 구성하므로,  
    `python train.py --config-name=train_diffusion_unet_hybrid_workspace task=lift_image_abs ...`  
    만으로 “DistilBERT + FiLM 통합된 Diffusion Policy”를 학습 가능


python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python eval.py \
  --checkpoint data/outputs/2026.02.02/19.46.05_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt \
  --output_dir data/pusht_eval_output \
  --device cuda:1


  
# Octo VLA github / https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py
예시 데이터(tfrecord) 다운 -> https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

# Diffusion VLA
예시 데이터(h5df) 다운 -> https://huggingface.co/datasets/lesjie/dexvla_example_data/blob/main/dexvla_example_data.zip

conda activate robodiff
python train.py \
  --config-name=train_diffusion_unet_dexvla_workspace \
  training.device=cuda:1

  