# PaddleMaterials

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-lightgrey" alt="English"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/简体中文-lightgrey" alt="简体中文"></a>
  <a href="README_ja.md"><img src="https://img.shields.io/badge/日本語-blue" alt="日本語"></a>
</p>

<p align="center"><img src="docs/ppmat_logo.png" alt="PaddleMaterials" width="400"></p>

<p align="center">
  <a href="Install.md"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://pypi.org/project/ppmat/"><img alt="PyPI バージョン" src="https://img.shields.io/pypi/v/ppmat?logo=pypi&amp;logoColor=white&amp;cacheSeconds=300"></a>
  <a href="LICENSE"><img alt="Apache 2.0 ライセンス" src="https://img.shields.io/github/license/PaddlePaddle/PaddleMaterials"></a>
  <a href="https://github.com/PaddlePaddle/PaddleMaterials/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMaterials?style=flat&amp;logo=github"></a>
</p>

## 🚀 はじめに

**PaddleMaterials** は、**PaddlePaddle** ディープラーニングフレームワークを基盤とするエンドツーエンドの AI4Materials ツールキットです。材料科学における基盤モデルの開発とデプロイを目的とした、データとメカニズムの二重駆動型プラットフォームとして設計されています。**PPMat** により、研究者は AI モデルを効率的に構築し、事前学習済みモデルを利用して材料探索を加速できます。

<p align="left"><img src="docs/overview_en.png" align="middle" width="1000"/></p>

### 🧩 コア機能

| タスク | 説明 | 主な用途 |
|--------|------|----------|
| **物性予測（PP）** | 構造から材料物性を予測 | 順設計、生成エネルギー・バンドギャップ・弾性率などの予測 |
| **構造生成（SG）** | 新しい結晶構造を生成 | 逆設計または構造生成 |
| **機械学習原子間ポテンシャル（MLIP）** | DFT の代理となる ML ポテンシャル | 分子動力学シミュレーション |
| **電子構造（ES）** | DFT の代理モデルとして物理場を予測 | 電子密度の予測 |
| **スペクトル解析（SE）** | スペクトルから構造を再構成 | NMR 構造解析 |
| **スペクトル強調（SPEN）** | 顕微鏡画像およびスペクトル信号を強調 | STEM 画像の強調、ノイズ除去 |

### 🧱 対応する材料

- **無機結晶** — 複数のデータセットと事前学習済みモデルによる充実したサポート
- **有機分子** — 低分子や一部のポリマーを含む複数のデータセットと事前学習済みモデルをサポート

### ✨ PaddleMaterials を選ぶ理由

- ✅ **豊富な事前学習済みモデルと AI-ready データセット** — 推論にすぐ利用できる 50 以上の事前学習済みモデルと、学習用に厳選された複数のデータセット
- ✅ **マルチタスク統合** — PP、SG、MLIP、ES、SE、SPEN などのタスクを統一されたフレームワークでサポート
- ✅ **マルチハードウェア対応** — NVIDIA GPU、MetaX GPU、Intel CPU を全面的にサポート
- ✅ **プロダクション対応** — 標準化された設計により使いやすく、分散学習、混合精度、チェックポイントからの再開に対応

### 📑 対応タスク

| タスク | 説明 | リンク |
|--------|------|--------|
| **物性予測（PP）** | 生成エネルギー、バンドギャップ、弾性特性を予測 | [README](property_prediction/README.md) |
| **構造生成（SG）** | 拡散モデルで新しい結晶構造を生成 | [README](structure_generation/README.md) |
| **機械学習原子間ポテンシャル（MLIP）** | 分子動力学向けの DFT 精度ポテンシャル | [README](interatomic_potentials/README.md) |
| **電子構造（ES）** | 電子構造特性を予測 | [README](electronic_structure/README.md) |
| **スペクトル解析（SE）** | NMR スペクトルから分子構造を再構成 | [README](spectrum_elucidation/README.md) |
| **スペクトル強調（SPEN）** | 顕微鏡画像とスペクトル信号を強調 | [README](spectrum_enhancement/README.md) |

### 🤖 利用可能な事前学習済みモデル

| タスク | モデル | データセット |
|--------|--------|--------------|
| **物性予測** | MEGNet、iComformer、DimeNet++、SphereNet | MP2018、MP2024、JARVIS、QM9 など |
| **構造生成** | MatterGen、DiffCSP | MP20、ALEX など |
| **機械学習原子間ポテンシャル** | CHGNet、MatterSim、SphereNet | MPTRJ、MD17 など |
| **電子構造** | InfGCN | QM9_ES、MP_ES、OMol25_MC_ES など |
| **スペクトル解析** | DiffNMR | MSD_NMR など |
| **スペクトル強調** | SFIN | SFIN-HAADF/BF など |

モデルの完全な一覧は [MODEL_REGISTRY](ppmat/models/__init__.py#L75) を参照してください。

---

## 🚀 クイックスタート

### 🔧 インストール

お使いのハードウェア環境に応じて、[インストールガイド](Install.md)を参照してください。マルチハードウェア対応の詳細については、[対応ハードウェア一覧](./docs/multi_device.md)を参照してください。

---

### ⚡ 簡単な推論

#### 物性予測

事前学習済み MEGNet モデルを使用して材料の生成エネルギーを予測します。

```bash
python property_prediction/predict.py \
    --model_name='megnet_mp2018_train_60k_e_form' \
    --weights_name='best.pdparams' \
    --input_format=cif --input_path='./property_prediction/example_data/cifs/' \
    --output_path='results/'
```

#### 構造生成

事前学習済み MatterGen モデルを使用して新しい結晶構造を生成します。

```bash
python structure_generation/sample.py \
    --model_name='mattergen_mp20' \
    --weights_name='latest.pdparams' \
    --output_path='result_mattergen_mp20/' \
    --mode='by_num_atoms' \
    --num_atoms=4
```

#### 原子間ポテンシャル

事前学習済み MatterSim モデルを使用してエネルギーと力を予測します。

```bash
python interatomic_potentials/predict.py \
    --model_name='mattersim_1M' \
    --weights_name='mattersim-v1.0.0-1M_model.pdparams' \
    --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/' \
    --output_path='results/'
```

#### 電子構造

事前学習済み InfGCN モデルを使用して、同梱のメタン例の電子密度を予測します。

```bash
python electronic_structure/predict.py \
    --model_name='infgcn_qm9' \
    --weights_name='best.pdparams' \
    --input_format=mol --input_path='electronic_structure/example_data/methane.mol' \
    --grid_shape=8 \
    --grid_batch_size=4096 \
    --output_path='output/infgcn_qm9/methane'
```

データセットまたはローカルチェックポイントを使用した推論については、
[InfGCN 予測ガイド](electronic_structure/configs/infgcn/README.md#prediction)を参照してください。

#### スペクトル解析

事前学習済み DiffNMR モデルと同梱の例を使用して NMR スペクトル解析を実行します。

```bash
python spectrum_elucidation/sample.py \
    --model_name='diffnmr_msdnmr_nless15' \
    --weights_name='best.pdparams' \
    --output_path='result_diffnmr_nless15/'
```

#### スペクトル強調

事前学習済み SFIN モデルを使用して STEM 画像を強調します。

```bash
python spectrum_enhancement/predict.py \
    --model_name='sfin_haadf_enhance' \
    --weights_name='best.pdparams' \
    --input_path='path/to/noisy_image.png' \
    --output_path='result_sfin/'
```

---

### 🏋️ 学習を始める

学習およびファインチューニングについては、[ドキュメント](get_started.md)を参照してください。

---

## 🤝 コントリビューター・協力組織・コミュニティ

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleMaterials&type=date&legend=top-left)](https://www.star-history.com/#PaddlePaddle/PaddleMaterials&type=date&legend=top-left)

PaddleMaterials の構築に貢献してくださったすべての皆様に感謝します！

<a href="https://github.com/PaddlePaddle/PaddleMaterials/graphs/contributors"><img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleMaterials" /></a>

ご協力いただいている以下の組織に感謝します！

<p align="left">
  <img src="docs/logo_SZNL_2.jpeg" align="middle" width="180"/>
  <img src="docs/logo_SinochemDI_2.jpeg" align="middle" width="180"/>
  <img src="docs/logo_MetaX.png" align="middle" width="140"/>
</p>

PaddleMaterials の WeChat グループに参加して、ぜひ交流してください！

<p align="left"><img src="docs/wechat_group.png" align="middle" width="100"/></p>

## 🛠️ PaddleMaterials への貢献

開発者の方は[アーキテクチャドキュメント](docs/ARCHITECTURE_ch.md)を参照してください。

---

## 📜 ライセンス

PaddleMaterials は [Apache License 2.0](LICENSE) の下で提供されています。

---

## 🎓 引用

```bibtex
@misc{liu2026paddlematerialsendtoenddeeplearning,
  title={PaddleMaterials: An End-to-End Deep Learning Toolkit for Materials Science},
  author={Xuwei Liu and Zhimin Zhang and Xiaoguang Hu and Zhongwei Liu and Yuanhui Sun and Jun Huo and Hua Wu and Haifeng Wang},
  year={2026},
  eprint={2611.03154},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.03154},
}
```

---

## 🙏 謝辞

本リポジトリは、以下のプロジェクトのコードを参考にしています。

[PaddleScience](https://github.com/PaddlePaddle/PaddleScience) |
[Matgl](https://github.com/materialsvirtuallab/matgl) |
[CDVAE](https://github.com/txie-93/cdvae) |
[DiffCSP](https://github.com/jiaor17/DiffCSP) |
[MatterGen](https://github.com/microsoft/mattergen) |
[MatterSim](https://github.com/microsoft/mattersim) |
[CHGNet](https://github.com/CederGroupHub/chgnet) |
[AIRS](https://github.com/divelab/AIRS)
