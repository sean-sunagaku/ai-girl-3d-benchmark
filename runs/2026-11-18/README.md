# GPT-6.2 Astra / GPT-6.2 Sol — 2026-11-18

共通の参照写真・CC0ベース・共通プロンプトを使った、GPT-6.2 Astra（`gpt-6.2-astra`）と
GPT-6.2 Sol（`gpt-6.2-sol`）の追加実行です。どちらも Codex CLI（0.167.2）のエージェント実行が、
初回（写真テクスチャ禁止）を制作しました。外見の採点・順位付けは行いません。

**今回はコーディネーターと参加モデルが別セッションです。** コーディネーター（Muse）が作業場所の準備と
起動・監視・検証を行い、制作は Codex CLI 上の各モデルが単独で行いました。実行条件の詳細は
[ハーネス記録](harness/README.md) にまとめています。

今回の指定により、サービスティアは通常の **Fast**（セッション記録では `priority`）を使い、
普段の Ultrafast は使っていません。指定方法と確認結果はハーネス記録に明記しています。

## GPT-6.2 Astra

- 初回: [Blenderシーン](procedural/gpt-6.2-astra/character.blend) / [閲覧用GLB](procedural/gpt-6.2-astra/character.glb) /
  [制作コード](procedural/gpt-6.2-astra/build.py) / [Astra の制作メモ](procedural/gpt-6.2-astra/README.md) /
  [コーディネーターの検査結果](procedural/gpt-6.2-astra/validation.json)
- 実行時間: 08:18:22 → 09:31:15 JST（4,373秒）、exit_code 0
- コスト: Codex 利用枠での実行（API直接課金なし）。トークン実績（入力 17,756,547＝うちキャッシュ 17,540,096、
  出力 106,112）から Fast の API レート（標準の2倍）で換算した推定で約 $25.98、Codex クレジット換算で約 325 credits。

| 元写真 | 初回の完成カット |
| --- | --- |
| ![元写真](../../benchmarks/cafe-selfie-v1/inputs/reference.png) | ![GPT-6.2 Astraの初回レンダー](procedural/gpt-6.2-astra/renders/beauty.png) |

| 正面 | 斜め | 横 | 背面 |
| --- | --- | --- | --- |
| ![正面](procedural/gpt-6.2-astra/renders/front.png) | ![斜め](procedural/gpt-6.2-astra/renders/three-quarter.png) | ![横](procedural/gpt-6.2-astra/renders/side.png) | ![背面](procedural/gpt-6.2-astra/renders/back.png) |

## GPT-6.2 Sol

- 初回: [Blenderシーン](procedural/gpt-6.2-sol/character.blend) / [閲覧用GLB](procedural/gpt-6.2-sol/character.glb) /
  [制作コード](procedural/gpt-6.2-sol/build.py) / [Sol の制作メモ](procedural/gpt-6.2-sol/README.md) /
  [コーディネーターの検査結果](procedural/gpt-6.2-sol/validation.json)
- 実行時間: 09:31:41 → 09:57:11 JST（1,530秒）、exit_code 0
- コスト: Codex 利用枠での実行（API直接課金なし）。トークン実績（入力 8,603,295＝うちキャッシュ 8,414,592、
  出力 93,196）から Fast の API レートで換算した推定で約 $4.30、Codex クレジット換算で約 54 credits。

| 元写真 | 初回の完成カット |
| --- | --- |
| ![元写真](../../benchmarks/cafe-selfie-v1/inputs/reference.png) | ![GPT-6.2 Solの初回レンダー](procedural/gpt-6.2-sol/renders/beauty.png) |

| 正面 | 斜め | 横 | 背面 |
| --- | --- | --- | --- |
| ![正面](procedural/gpt-6.2-sol/renders/front.png) | ![斜め](procedural/gpt-6.2-sol/renders/three-quarter.png) | ![横](procedural/gpt-6.2-sol/renders/side.png) | ![背面](procedural/gpt-6.2-sol/renders/back.png) |

- [実行条件](run.json) / [ハーネス記録](harness/README.md) / [検証スクリプト](harness/coordinator_validate.py)

## この実行の目標と達成状況（両モデル共通）

| 完了条件 | 結果 |
| --- | --- |
| 共通入力と課題文のSHA-256がマニフェストと一致、作業場所の `PROMPT.md` が共通原文とバイト一致 | 一致（各 validation.json） |
| 課題文の成果物（`build.py`、`character.blend`、5視点のPNG、`README.md`、`scene-report.json`）が揃う | 揃っている |
| 独立コピーでの再実行が完走する（Astra: `build.py`→`render.py`→`report.py`、Sol: `build.py`→`render.py`→`report_scene.py`） | 両モデルとも全ステップ終了コード0（Astra 147.1秒、Sol 91.3秒） |
| 再生成した5画像の画素データ（IDAT）が元レンダーと一致 | 両モデルとも全5枚一致 |
| Cycles CPU・3スレッド・32サンプル・デノイズ、テクスチャ2048以下、GPU不使用 | 検査で確認（各 validation.json） |
| 完成カット960×1040、正面・斜め・横・背面768×768 | 全5枚が指定寸法 |
| 保存シーンを保存し直さずに再オープンし、SHA-256が変わらない | 不変（各 validation.json） |
| 人物だけのGLBを書き出し、再インポートできる | 成功（Astra: 18メッシュ、Sol: 15メッシュ） |
| 収録ファイルのプライバシースキャン | 秘密情報は両モデルとも検出なし。Astra はスキャン自体もクリーン。Sol は `character.blend` と各PNGのメタデータ（レンダー時のFileパス）に Mac のユーザー名を含む絶対パスが残っている。これは Kimi・Terra・Carbon の公開済み収録物にも同様に含まれているもので、今回も制作物そのままを収録した（詳細は run.json の privacy）。 |

## 追加版について

写真テクスチャ許可版（第2段階）は今回未実施です。実施するかどうかは本人の判断で決めます。

## 成果物の限界

両モデルの制作メモにあるとおり、見てCGと分かる仕上がりで、写真の人物の再現ではありません。
顔はCC0ベースを手続き的に変形したもので、横顔・背面・手などは1枚の写真からの推定です。
制作向けのリグや下半身は含まれません。
