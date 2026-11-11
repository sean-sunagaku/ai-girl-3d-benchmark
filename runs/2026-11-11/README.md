# Gemini 4 Carbon — 2026-11-11

共通の参照写真・CC0ベース・共通プロンプトを使った、Gemini 4 Carbon（`gemini-4-carbon`）の追加実行です。
Gemini CLI（0.68.0）のエージェント実行が、初回（写真テクスチャ禁止）を制作しました。
外見の採点・順位付けは行いません。

**今回はコーディネーターと参加モデルが別セッションです。** コーディネーター（Muse）が作業場所の準備と
起動・監視・検証を行い、制作は Gemini CLI 上の Carbon が単独で行いました。実行条件の詳細は
[ハーネス記録](harness/README.md) にまとめています。

- 初回: [Blenderシーン](procedural/gemini-4-carbon/character.blend) / [閲覧用GLB](procedural/gemini-4-carbon/character.glb) /
  [制作コード](procedural/gemini-4-carbon/build.py) / [Carbon の制作メモ](procedural/gemini-4-carbon/README.md) /
  [コーディネーターの検査結果](procedural/gemini-4-carbon/validation.json)
- [実行条件](run.json) / [ハーネス記録](harness/README.md) / [検証スクリプト](harness/coordinator_validate.py)

## 初回：写真由来の人物テクスチャは禁止

| 元写真 | 初回の完成カット |
| --- | --- |
| ![元写真](../../benchmarks/cafe-selfie-v1/inputs/reference.png) | ![Gemini 4 Carbonの初回レンダー](procedural/gemini-4-carbon/renders/beauty.png) |

| 正面 | 斜め | 横 | 背面 |
| --- | --- | --- | --- |
| ![正面](procedural/gemini-4-carbon/renders/front.png) | ![斜め](procedural/gemini-4-carbon/renders/three-quarter.png) | ![横](procedural/gemini-4-carbon/renders/side.png) | ![背面](procedural/gemini-4-carbon/renders/back.png) |

## この実行の目標と達成状況

| 完了条件 | 結果 |
| --- | --- |
| 共通入力5件と課題文のSHA-256がマニフェストと一致、作業場所の `PROMPT.md` が共通原文とバイト一致 | 一致（validation.json） |
| 課題文の成果物（`build.py`、`character.blend`、5視点のPNG、`README.md`、`scene-report.json`）が揃う | 揃っている |
| `Blender --background --factory-startup --threads 3 --python build.py` が入力から完走する | 独立コピーでの再実行が終了コード0（約180秒） |
| Cycles CPU・3スレッド・32サンプル・デノイズ、テクスチャ2048以下、評価頂点総計100万未満、GPU不使用 | 検査で確認（scene-report.json と validation.json） |
| 完成カット960×1040、正面・斜め・横・背面768×768 | 全5枚が指定寸法 |
| 保存シーンを保存し直さずに再オープンし、SHA-256が変わらない | 不変（validation.json） |
| 人物だけのGLBを書き出し、再インポートできる | 成功（validation.json） |
| 収録ファイルのプライバシースキャン | 秘密情報（APIキー・メールアドレス等）は検出なし。一方、`character.blend` と各PNGのメタデータ（レンダー時のFileパス）に Mac のユーザー名を含む絶対パスが残っている。これは Kimi・Terra の公開済み収録物にも同様に含まれているもので、今回も制作物そのままを収録した（Fable 実行だけは参加モデル自身が build.py で除去していた）。詳細は validation.json の privacy_scan |

Fable 5.2 実行との違いとして、再ビルドで再生成した5画像の画素データ（IDAT）は元レンダーと
完全一致しませんでした（サイズ差0.1%未満）。再ビルド自体は正常に完走しており、
寸法とファイルサイズは一致しています。この点も validation.json に記録しています。

## 追加版について

写真テクスチャ許可版（第2段階）は今回未実施です。実施するかどうかは本人の判断で決めます。

## 成果物の限界

Carbon の制作メモにあるとおり、見てCGと分かる仕上がりで、写真の人物の再現ではありません。
顔はCC0ベースを手続き的に変形したもので、横顔・背面・手などは1枚の写真からの推定です。
制作向けのリグや下半身は含まれません。
