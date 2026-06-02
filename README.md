# Europa Surface Biosignature-Candidate ML Triage: Experiment Starter

이 폴더는 유로파 표면 분광-공간 맥락 기반 biosignature-candidate triage 논문의 1차 실험을 바로 시작하기 위한 최소 코드 뼈대입니다.

## 핵심 원칙

- 메인 파장 범위: Galileo/NIMS 기준 0.7--5.2 µm.
- 라벨은 feature 조합으로 만들지 않습니다. 숨겨진 생성 원인 `z`에서 라벨을 만듭니다.
- 모델은 `z`를 보지 않고 noisy spectrum + geology/radiation proxy만 봅니다.
- 메인 평가는 `ocean_organic` vs rest ranking 문제입니다.
- 실제 Galileo/NIMS 적용은 validation이 아니라 qualitative sanity check입니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

선택적으로 GitHub/외부 패키지:

```bash
# PDS4 자료 읽기
pip install pds4-tools

# hyperspectral cube/ENVI 자료 처리
pip install spectral

# USGS Spectral Library v7 local archive loader
pip install 'splib07-loader @ git+https://github.com/brianschubert/splib07-loader.git'
```

## 실행 순서

```bash
# 1. synthetic dataset 생성
python scripts/01_generate_dataset.py --n 8000 --rho-geo 0.75 --rho-rad 0.75

# 2. Experiment 1: spectral-only vs spatial-spectral
python scripts/02_run_experiment1.py

# 3. Experiment 2: prior-strength sweep
python scripts/03_prior_sweep.py

# 4. Experiment 3: same-spectrum/different-location stress test
python scripts/04_same_spectrum_test.py
```

결과는 `results/`에 CSV/PNG로 저장됩니다.

## 이후 실제 자료 연결

1. `data/endmembers/`에 실험실 spectra CSV를 넣습니다. 형식은 `wavelength_um, reflectance` 또는 `wavelength_um, intensity`입니다.
2. `src/loaders.py`의 placeholder 함수를 사용해 NIMS/RELAB/USGS/PAHdb 자료를 로드합니다.
3. `src/endmembers.py`의 toy Gaussian endmember를 실제 spectrum interpolation으로 교체합니다.
4. Galileo/NIMS real-data sanity check는 별도 script로 추가합니다. ground truth가 없으므로 validation이라고 쓰면 안 됩니다.

## 출력 파일

- `results/experiment1_metrics.csv`
- `results/experiment1_pr_curve.png`
- `results/prior_sweep_metrics.csv`
- `results/prior_sweep_heatmap_fpr_reduction.png`
- `results/same_spectrum_scores.csv`
- `results/same_spectrum_comparison.png`
