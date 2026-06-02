# Europa Surface Biosignature-Candidate ML Triage 실험 최종 정리

작성일: 2026-06-02

## 1. 목적

이 실험은 Europa 표면의 NIMS 유사 반사 스펙트럼과 지질/방사선 문맥 정보를 함께 사용했을 때, `ocean_organic` 후보를 단순 스펙트럼 기반 모델보다 더 잘 우선순위화할 수 있는지 확인하기 위한 synthetic triage 실험이다.

핵심 질문은 다음과 같다.

- 스펙트럼만 사용하는 모델이 방사선 변성 물질이나 외부 기원 유기물 같은 hard negative를 과대평가하는가?
- 지질 문맥과 방사선 문맥을 추가하면 `ocean_organic` 후보 랭킹이 개선되는가?
- 문맥 정보만으로는 충분하지 않고, 스펙트럼과 결합될 때 유용한가?

이 결과는 실제 Galileo/NIMS 검증 결과가 아니라, 실험 설계와 synthetic data에서의 sanity check이다.

## 2. 실험 설계

### 2.1 예측 문제

- 이진 랭킹 문제: `ocean_organic` vs rest
- 양성 클래스: `ocean_organic`
- 음성 클래스:
  - `ocean_nonorganic`
  - `radiation_mimic`
  - `exogenic_complex_organic`
  - `noise_artifact`
- 평가는 top 10% 후보 랭킹에서 precision, recall, hard negative 유입량을 중심으로 수행했다.

### 2.2 입력 feature 설정

모델 비교는 5개 feature setting으로 수행했다.

| setting | 사용 feature |
|---|---|
| `spectral_only` | NIMS 유사 스펙트럼 bin만 사용 |
| `spectral_geology` | 스펙트럼 + 지질 문맥 |
| `spectral_radiation` | 스펙트럼 + 방사선 문맥 |
| `full` | 스펙트럼 + 지질 문맥 + 방사선 문맥 |
| `context_only` | 지질 문맥 + 방사선 문맥만 사용 |

지질 문맥 feature:

- `chaos_proximity`
- `lineament_proximity`
- `ridge_proximity`
- `young_terrain`
- `activity_proxy`

방사선 문맥 feature:

- `trailing_hemisphere`
- `radiation_exposure`
- `sulfur_proxy`
- `rad_mimic_proxy`

### 2.3 스펙트럼 범위와 해상도

- 파장 범위: 0.7-5.2 um
- 간격: 0.025 um
- bin 수: 181
- noise preset: `moderate`
  - Gaussian noise sigma: 0.020
  - smoothing window: 5

## 3. 데이터

### 3.1 Synthetic dataset

최종 실험은 `generate_dataset()`로 생성한 synthetic sample을 사용했다.

- 기본 sample 수: seed별 `n=8000`
- train/test split: 70/30
- stratified by target `y`
- hidden class prior:
  - `ocean_organic`: 0.18
  - `ocean_nonorganic`: 0.22
  - `radiation_mimic`: 0.28
  - `exogenic_complex_organic`: 0.20
  - `noise_artifact`: 0.12

생성된 주요 데이터 파일:

- `data/processed/synthetic_dataset.csv`
- `data/processed/synthetic_dataset_lab_endmember_v1.csv`
- `data/processed/qc_smoke_lab_endmember_v2.csv`

### 3.2 Endmember library

현재 endmember는 일부 RELAB 기반/일부 parametric proxy 기반이다. 모든 endmember는 0.7-5.2 um, 181 bin으로 보간되어 있다.

| endmember | source | 비고 |
|---|---|---|
| `ice` | parametric proxy | 물 얼음 proxy |
| `ocean_salt` | hybrid proxy | RELAB `c1cc15` + hydration/salt proxy |
| `simple_organic` | parametric proxy | C-H, C=O/carboxyl feature proxy |
| `tholin_pah` | parametric proxy | PAH/tholin hard-negative proxy |
| `sulfuric_acid_hydrate` | parametric proxy | radiolysis proxy |
| `sulfur_so2` | parametric proxy | weak sulfur/SO2 proxy |
| `h2o2` | parametric proxy | 3.5 um H2O2-like feature proxy |
| `rad_salt_proxy` | derived proxy | salt + sulfuric-acid-hydrate mixture proxy |

검증 파일:

- `results/processed_endmember_validation.csv`
- `results/processed_endmembers.png`
- `data/manifest/endmember_selection.csv`

주의: 이 endmember set은 실제 biosignature molecule 검출용 ground truth가 아니다. 실험의 목적은 "문맥 결합이 hard-negative ranking을 줄이는가"를 보는 것이다.

## 4. 모델과 평가

### 4.1 모델

최종 메인 결과는 Random Forest를 사용했다.

- model: `rf`
- estimator: `RandomForestClassifier`
- trees: 150
- max depth: 18
- min samples leaf: 3
- class weight: `balanced_subsample`
- calibration: isotonic calibration with 3-fold CV

비교/보조 실험으로 logistic regression 결과도 일부 생성되어 있으나, 최종 요약은 `rf` 20-seed 결과를 기준으로 한다.

### 4.2 평가 지표

주요 지표:

- PR-AUC
- ROC-AUC
- Precision at top 10%
- Recall at top 10%
- Brier score
- Top 10% 안에 들어온 `radiation_mimic` 개수
- Top 10% 안에 들어온 `exogenic_complex_organic` 개수

## 5. 최종 결과: Experiment 1, RF 20 seeds

최종 결과 파일:

- `results/experiment1_metrics_rf_20seed.csv`
- `results/experiment1_metrics_by_seed_rf_20seed.csv`

| setting | PR-AUC | ROC-AUC | Precision@10% | Recall@10% | Brier | top10 radiation mimic |
|---|---:|---:|---:|---:|---:|---:|
| `spectral_only` | 0.9503 +/- 0.0075 | 0.9863 +/- 0.0025 | 0.9896 +/- 0.0055 | 0.5566 +/- 0.0145 | 0.0325 +/- 0.0026 | 0.3500 +/- 0.4894 |
| `spectral_geology` | 0.9649 +/- 0.0055 | 0.9909 +/- 0.0017 | 0.9952 +/- 0.0028 | 0.5694 +/- 0.0226 | 0.0265 +/- 0.0022 | 0.0000 +/- 0.0000 |
| `spectral_radiation` | 0.9572 +/- 0.0069 | 0.9885 +/- 0.0021 | 0.9919 +/- 0.0039 | 0.5575 +/- 0.0125 | 0.0297 +/- 0.0026 | 0.0000 +/- 0.0000 |
| `full` | 0.9661 +/- 0.0052 | 0.9913 +/- 0.0017 | 0.9952 +/- 0.0034 | 0.5716 +/- 0.0255 | 0.0262 +/- 0.0024 | 0.0000 +/- 0.0000 |
| `context_only` | 0.4486 +/- 0.0171 | 0.8644 +/- 0.0058 | 0.4492 +/- 0.0307 | 0.3929 +/- 0.1756 | 0.0999 +/- 0.0024 | 0.0000 +/- 0.0000 |

### 해석

`full` 모델이 전체적으로 가장 좋은 성능을 보였다.

- PR-AUC: `spectral_only` 0.9503 -> `full` 0.9661
- Recall@10%: `spectral_only` 0.5566 -> `full` 0.5716
- Brier: `spectral_only` 0.0325 -> `full` 0.0262
- top10 radiation mimic: `spectral_only` 평균 0.35개 -> `full` 0개

문맥만 쓰는 `context_only`는 PR-AUC 0.4486으로 충분하지 않았다. 즉, 지질/방사선 문맥은 단독 검출기가 아니라 스펙트럼 ambiguity를 줄이는 보조 정보로 해석하는 것이 맞다.

## 6. Prior-strength sweep

결과 파일:

- `results/prior_sweep_metrics.csv`
- `results/prior_sweep_heatmap_fpr_reduction.png`
- `results/prior_sweep_heatmap_delta_pr_auc.png`
- `results/prior_sweep_heatmap_delta_recall10.png`
- `results/prior_sweep_heatmap_top10_rad_reduction.png`

실험은 `rho_geo`, `rho_rad`를 0.0-1.0 범위에서 바꾸며, 문맥 prior가 약할 때와 강할 때의 효과를 비교했다.

전체 25개 조건 평균:

- 평균 delta PR-AUC (`full - spectral_only`): +0.0076
- 평균 delta Recall@10%: +0.0242
- 평균 radiation mimic FPR reduction: +0.0007
- 평균 top10 radiation mimic reduction: +0.16개

가장 큰 PR-AUC 개선 조건:

| rho_geo | rho_rad | spectral PR-AUC | full PR-AUC | delta PR-AUC |
|---:|---:|---:|---:|---:|
| 1.0 | 0.5 | 0.9418 | 0.9699 | +0.0282 |
| 0.75 | 0.0 | 0.9516 | 0.9720 | +0.0204 |
| 0.75 | 0.75 | 0.9778 | 0.9929 | +0.0151 |
| 0.75 | 0.5 | 0.9655 | 0.9806 | +0.0150 |
| 1.0 | 0.25 | 0.9762 | 0.9912 | +0.0150 |

가장 큰 Recall@10% 개선 조건:

| rho_geo | rho_rad | spectral recall@10% | full recall@10% | delta recall |
|---:|---:|---:|---:|---:|
| 0.75 | 0.0 | 0.5704 | 0.7333 | +0.1630 |
| 1.0 | 0.0 | 0.7188 | 0.8203 | +0.1016 |
| 0.5 | 0.0 | 0.5758 | 0.6742 | +0.0985 |
| 0.75 | 0.75 | 0.6000 | 0.6929 | +0.0929 |
| 1.0 | 1.0 | 0.7287 | 0.8062 | +0.0775 |

### 해석

문맥 prior가 항상 성능을 크게 올리지는 않는다. 하지만 `rho_geo`가 중간 이상일 때는 full 모델이 spectral-only보다 일관되게 유리한 조건이 많았다. 특히 top 10% 후보 랭킹에서 recall과 radiation mimic 제거 효과가 함께 나타났다.

## 7. Same-spectrum / different-location stress test

결과 파일:

- `results/same_spectrum_paired_stats_rf_ambiguous.csv`
- `results/same_spectrum_paired_deltas_rf_ambiguous.csv`
- `results/same_spectrum_scores_rf_ambiguous.csv`
- `results/same_spectrum_comparison_rf_ambiguous.png`

설계:

- 동일한 또는 매우 유사한 spectrum을 가진 쌍을 만든다.
- 한쪽은 더 plausible한 문맥, 다른 쪽은 덜 plausible한 문맥으로 배치한다.
- spectral-only와 full 모델의 score 차이를 비교한다.

RF ambiguous stress test 결과:

- pair 수: 200
- ambiguous source score range: 0.4-0.7
- spectral score delta mean: 0.0000
- full score delta mean: 0.6045
- full score delta std: 0.2597
- sign test p-value: 1.24e-60

### 해석

스펙트럼만 보면 구분 불가능한 쌍에서 spectral-only 모델은 score 차이를 만들지 못했다. 반면 full 모델은 위치/문맥 차이를 반영해 평균 0.6045의 score 차이를 만들었다. 이는 "동일 스펙트럼이라도 Europa 표면 문맥이 후보 우선순위에 영향을 줘야 한다"는 실험 가설과 부합한다.

## 8. 결론

현재 synthetic 실험 기준 결론은 다음과 같다.

1. 스펙트럼 단독 모델도 PR-AUC가 높지만, hard negative ambiguity가 남는다.
2. 지질/방사선 문맥을 결합한 `full` 모델은 PR-AUC, ROC-AUC, calibration, top10 recall을 개선했다.
3. `full` 모델은 top10 후보 안의 `radiation_mimic` 유입을 20-seed 평균 기준 0으로 줄였다.
4. `context_only`는 성능이 낮아, 문맥은 독립 검출기가 아니라 스펙트럼 기반 후보를 재정렬하는 prior로 써야 한다.
5. Same-spectrum stress test에서 full 모델은 동일 스펙트럼 후보의 위치 문맥 차이를 강하게 반영했다.

따라서 이 실험은 "Europa biosignature-candidate triage에서는 spectrum-only ranking보다 spatial-spectral ranking이 더 안전한 후보 우선순위화 전략일 수 있다"는 주장을 synthetic setting에서 지지한다.

## 9. 한계와 다음 단계

한계:

- 실제 Galileo/NIMS 관측 검증이 아니다.
- endmember 중 다수가 parametric proxy이며, 직접 실험실 스펙트럼이 아니다.
- hidden class와 문맥 feature의 관계는 설계자가 부여한 synthetic prior다.
- 404/다운로드 문제로 RELAB metadata 수집은 일부만 완료된 상태다.
- 모델 성능은 실제 생명징후 검출 성능이 아니라, synthetic ranking task의 성능이다.

다음 단계:

- RELAB/USGS/PAHdb 기반 endmember를 더 실제적인 스펙트럼으로 교체한다.
- Galileo/NIMS footprint 또는 global map proxy와 연결한 qualitative sanity check를 추가한다.
- `same-spectrum` stress test를 고정된 synthetic pair뿐 아니라 실제 유사 스펙트럼 후보군에도 적용한다.
- 문맥 prior strength를 실제 Europa geology/radiation map의 uncertainty와 연결한다.
- RF 외에 calibrated HGB/logistic baseline을 동일 seed 수로 정리한다.

## 10. 주요 산출물

메인 결과:

- `results/experiment1_metrics_rf_20seed.csv`
- `results/experiment1_metrics_by_seed_rf_20seed.csv`

Prior sweep:

- `results/prior_sweep_metrics.csv`
- `results/prior_sweep_heatmap_fpr_reduction.png`
- `results/prior_sweep_heatmap_delta_pr_auc.png`
- `results/prior_sweep_heatmap_delta_recall10.png`
- `results/prior_sweep_heatmap_top10_rad_reduction.png`

Stress test:

- `results/same_spectrum_paired_stats_rf_ambiguous.csv`
- `results/same_spectrum_paired_deltas_rf_ambiguous.csv`
- `results/same_spectrum_scores_rf_ambiguous.csv`
- `results/same_spectrum_comparison_rf_ambiguous.png`

Endmember/QC:

- `results/processed_endmember_validation.csv`
- `results/processed_endmembers.png`
- `data/manifest/endmember_selection.csv`
