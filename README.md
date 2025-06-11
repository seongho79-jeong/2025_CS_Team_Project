# 2025_CS_Team_Project


# Fourier‑Phase Jitter with Graph Consistency (FPJ‑GC)

**개요**  
Phase 스펙트럼에 저·고주파별 미세 노이즈를 가해 *스타일‑불변 구조 변형*을 만들고,  
원본 배치와 Jitter 배치가 **동일한 유사도 그래프**를 유지하도록 학습 ⇒  
도메인 변화(Art, Cartoon, Sketch ↔ Photo)에 강인한 Feature 획득.

---

## 1. 핵심 아이디어

| 구성 | 역할 | 효과 |
|------|------|------|
| **Phase Jitter** | θ<sub>L</sub> ← θ<sub>L</sub> + ε<sub>L</sub>,  θ<sub>H</sub> ← θ<sub>H</sub> − ε<sub>H</sub> | 구조·윤곽을 살짝 이동 → 도메인 shape shift 시뮬레이션 |
| **Graph Consistency** | 배치 내 cos‑sim 행렬 S vs S̃ 유지 (KL/MSE) | 클래스 간 *상대적 거리* 보존 → Jitter 뷰와 표현 정렬 |

> Amplitude(MixStyle·FACT) 와 달리 **Phase** 조작,  
> 개별 샘플 일치 대신 **배치‑전역 그래프** 일관성으로 안정적 수렴.

---

## 2. 알고리즘

```text
for each batch X (B,3,H,W):
    1) FFT → A, Θ
    2) Split Θ into low & high freq (radius r_split)
       Θ̃ = Θ + ε_L·mask_L – ε_H·mask_H
    3) iFFT → X̃  (Phase‑Jitter view)
    4) Forward:
         z  = f(X),    logits  = head(z)
         z̃ = f(X̃),   logits̃ = head(z̃)
    5) Loss:
       L_cls   = CE(logits, y) + CE(logits̃, y)
       S   = softmax(z·zᵀ/τ)
       S̃  = softmax(z̃·z̃ᵀ/τ)
       L_g = KL(S‖S̃) + KL(S̃‖S)
       L   = L_cls + λ_g · L_g
```

---

## 3. 주요 하이퍼

| 파라미터 | 기본 | 설명 |
|----------|------|------|
| `σ_L / σ_H` | 0.05 / 0.10 | 저·고주파 jitter 세기 |
| `r_split` | 0.3 (Nyquist 비율) | 저/고 경계 |
| `λ_g` | 0.2 | 그래프 일관성 가중치 |
| `τ`  | 0.07 | similarity temperature |

*Tip*: 학습 후기 `σ`·`λ_g` 를 줄여 과도 수렴 방지.

---

## 4. 특징 & 장점

* **Phase 중심 변조** → shape‑shift 도메인에 강함 (Sketch, Cartoon)  
* **Batch‑level 그래프** → 개별 alignment 보다 안정, collapse 위험 ↓  
* 추가 파라미터 0 (FFT 연산만)  
* 입력/배치/epoch 제약 無 → 과제 코드 최소 수정

---

## 5. 기존 기법과 차별성

| 기법 | 조작 | 일관성 손실 | 차이 |
|------|------|-------------|------|
| FACT | Amplitude Mix | KL(o↔mix) | **Phase** 변조, 그래프(전체) |
| MixStyle | 채널‑µ,σ 교환 | 없음 | spatial‑free vs Phase‑freq |
| FD‑SDG | Amplitude Drop | 없음 | jitter, +Graph consistency |

DG 문헌에 **Phase‑Jitter + Similarity Graph** 조합은 아직 드뭄.

---

## 6. 코드 스니펫 (64×64 예)

```python
def phase_jitter(imgs, r=0.3, sL=0.05, sH=0.1):
    B,C,H,W = imgs.shape
    fft = torch.fft.rfft2(imgs, norm='ortho')
    amp, ph = torch.abs(fft), torch.angle(fft)

    fy = torch.fft.fftfreq (H, d=1./H).to(imgs.device)
    fx = torch.fft.rfftfreq(W, d=1./W).to(imgs.device)
    R  = torch.sqrt(fy[:,None]**2 + fx[None,:]**2)

    maskL = (R <= r); maskH = ~maskL
    ph_j  = ph + sL*torch.randn_like(ph)*maskL - sH*torch.randn_like(ph)*maskH
    return torch.fft.irfft2(amp*torch.exp(1j*ph_j), s=(H,W), norm='ortho')
```

---

## 7. 성능 기대

| 모델 | PACS Photo Acc. |
|------|-----------------|
| ResNet‑50 baseline | 63 % ±1 |
| **FPJ‑GC (본 연구)** | **66–68 %** |

Phase 와 Graph 두 축 모두 고려하여 스타일·형태 도메인 불일치를 완화합니다.


# Latent‑Patch Re‑Masking (LaPaR)

**한 줄 개념**  
학습 배치 안에서 *서로 다른 도메인* 이미지의 **중간‑feature 패치**를 교환(교차 이식)하고,  
① 교환된 패치가 **원본 클래스**를 유지하도록 CE 손실  
② **원본 ↔ 재구성 일관성**을 추가 L<sub>recon</sub> 으로 강제  
→ 국지·전역 도메인 편향을 모두 희석해 Generalization 향상.

---

## 1. 왜 ‘패치‑교환’인가?

| 레벨 | 기존 방식 | 한계 | LaPaR 해결 |
|------|-----------|------|------------|
| 입력 이미지 | CutMix, StyleMix | shape·texture 모두 뒤섞여 라벨 혼동 위험 | feature 공간이므로 semantics 유지 |
| 주파수 | FACT, F‑Drop | global shift 위주, local texture 놓침 | spatial patch 단위 편향 제거 |
| **Feature patch (LaPaR)** | – | – | class 유지 + 도메인 노이즈 주입 |

---

## 2. 알고리즘 요약
```text
for each batch X (B,3,H,W):
    1) layer L 출력 → F = f_L(X)    # (B,C,h,w)
    2) K개 패치 좌표 {(i,j)} 선택
    3) 도메인 다르게 → 패치 교환: F'[b,:,i,j] ← F[b',:,i,j]
    4) forward 계속 → logits_aug
    5) Loss
       L_cls   = CE(logits_aug, y)
       L_recon = ‖g_L(F') – X‖₁
       L_total = L_cls + α·L_recon
```

---

## 3. 주요 하이퍼
| 하이퍼 | 기본값 | 설명 |
|--------|--------|------|
| Patch 크기 `p` | 4 (latent cell) | 너무 크면 CutMix 비슷 |
| 교환 비율 `K_frac` | 0.3 | 전체 cell 의 30 % |
| 재구성 가중 `α` | 0.2 | 클래스 vs 복원 균형 |
| 적용 레이어 `L` | ResNet **layer3** | 14×14 해상도, 표현력 ↑ |

---

## 4. 손실 해석
1. **L<sub>cls</sub>** – 교환 후에도 같은 라벨 → 패치의 도메인 정보가 분류와 무관  
2. **L<sub>recon</sub>** – 디코더가 원본 스타일 복원 → 네트워크가 스스로 편향 보정

---

## 5. 특징 & 장점
* 도메인‑국지 texture 랜덤화  
* Consistency / Adversarial 필요 없음 → 구현 간단  
* 추가 파라미터 <0.5 M (디코더), 테스트 시 제거 가능  
* 과제 조건(64×64, 32batch, 10epoch) 그대로 사용

---

## 6. 기존 연구와 차별성
| 기존 | 차이 |
|------|------|
| CutMix·PatchMix | 이미지 & 라벨 Mix → 라벨 유지 불가 |
| MixStyle | 채널‑통계 교환 → LaPaR 는 공간‑patch 교환 |
| MAE | 자기 복원만 → LaPaR 는 **교차 도메인** + 분류 |

DG 문헌에서 *feature‑patch cross‑domain 교환 + Reconstruction* 조합은 아직 희소.

---

## 7. PyTorch 구현 핵심
```python
def remask_patch(F, dom_ids, k_frac=0.3):
    B,C,h,w = F.size()
    num = int(k_frac * h * w)
    idx = torch.randperm(h*w, device=F.device)[:num]
    i, j = idx // w, idx % w

    perm = torch.randperm(B, device=F.device)
    swap = dom_ids != dom_ids[perm]
    perm = torch.where(swap, perm, torch.roll(perm, 1))

    F_aug = F.clone()
    for b in range(B):
        F_aug[b, :, i, j] = F[perm[b], :, i, j]
    return F_aug
```

> **PACS baseline** 대비 **+3 – 5 pp** 성능 향상을 기대할 수 있습니다.
