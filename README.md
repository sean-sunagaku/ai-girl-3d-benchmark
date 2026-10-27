# AI Girl 3D Benchmark

参照写真1枚から、2026-10-14の4モデルと、後日追加したKimi K3.1 Pro・GPT-6.1 Terraが実際のBlenderで制作した人物の上半身モデルを公開します。元の写真、渡した指示、生成コード、レンダー画像、編集可能なBlenderシーンを、誰でも見て使える形で収録しています。

**外見の採点・順位付けは行いません。元写真と成果物を見比べてください。**

- [2026-10-14の40画像と追加実行を見る](GALLERY.md)
- [元の参照写真を開く](benchmarks/cafe-selfie-v1/inputs/reference.png)
- [2026-10-14のBlenderシーン一式・Web用GLBをダウンロード](https://github.com/sean-sunagaku/ai-girl-3d-benchmark/releases/tag/as-published-2026-10-15)
- [同じ入力で制作する・既存のモデルを開く](docs/REPRODUCE.md)
- [実験条件と収録範囲](docs/PROTOCOL.md)
- [AI向けの制作・閲覧手順スキル](skills/blender-character-benchmark/SKILL.md)

## GPT-6.1 Terra の追加実行（2026-10-27）

同じ元写真・共通ベース・初回プロンプトを使ったTerraの結果を収録しました。
[制作記録と5方向の画像](runs/2026-10-27/README.md)、
[編集可能なBlenderシーン](runs/2026-10-27/procedural/gpt-6.1-terra/character.blend)、
[閲覧用GLB](runs/2026-10-27/procedural/gpt-6.1-terra/character.glb) を取得できます。
対話版Codex CLIでの制作、SSH切断後の継続指示、コーディネーターによる再実行・検査は
[今回の記録](runs/2026-10-27/README.md) に明記しています。完成したのは初回の写真テクスチャ禁止版です。

## Kimi K3.1 Pro の追加実行（2026-10-25）

同じ元写真・共通ベース・初回プロンプトを使ったKimiの追加結果を収録しました。
[制作記録と5方向の画像](runs/2026-10-25/README.md)、
[編集可能なBlenderシーン](runs/2026-10-25/procedural/kimi-k3.1-pro/character.blend)、
[閲覧用GLB](runs/2026-10-25/procedural/kimi-k3.1-pro/character.glb) を取得できます。

[HTMLビューアー](viewer/index.html)ではTerraとKimiを切り替え、元写真、回転・拡大できるGLB、各5方向のPNGを表示します。
リポジトリのルートで `python3 -m http.server 8765 --bind localhost` を実行し、
`http://localhost:8765/viewer/` を開いてください。手順は [viewer/README.md](viewer/README.md) にあります。

**完成しているのは初回の写真テクスチャ禁止版です。写真テクスチャ許可版はAPIの日次トークン上限で未実行です。**
OpenCodeでの継続指示、コード回収、Blender API互換修正、コーディネーターによる実行・検査は
[今回の記録](runs/2026-10-25/README.md) に明記しています。追加課金・他モデルの再実行は行っていません。
以下の4モデル・2026-10-14の公開済み成果物は、そのまま保持しています。

## 元の参照写真

<img src="benchmarks/cafe-selfie-v1/inputs/reference.png" alt="茶色の長い髪とピンクのニットの人物を写した参照写真" width="420">

このPNGは、実験で使った1205×1306ピクセルの元ファイルをそのまま収録しています。リサイズ・再圧縮していません。画像本体を通常のGitで管理しているため、cloneすれば取得できます。

## 4モデル × 2段階

2026年10月14日の実験です。表のモデル名・IDは当時の実行記録に基づきます。

| モデル | 指定ID | ハーネス | 初回コード・画像 | 写真テクスチャ許可版 |
| --- | --- | --- | --- | --- |
| Gemini 4 Argon High | `gemini-4-argon-high` | Cursor CLI | [開く](runs/2026-10-14/procedural/argon) | [開く](runs/2026-10-14/photo-textured/argon) |
| GPT-6.1 Sol | `gpt-6.1-sol` | Codex CLI | [開く](runs/2026-10-14/procedural/gpt-6.1-sol) | [開く](runs/2026-10-14/photo-textured/gpt-6.1-sol) |
| Claude Opus 5.5 | `claude-opus-5-5` | Claude Code | [開く](runs/2026-10-14/procedural/opus-5.5) | [開く](runs/2026-10-14/photo-textured/opus-5.5) |
| Grok 4.7 High | `grok-4.7-high` | Cursor CLI | [開く](runs/2026-10-14/procedural/grok) | [開く](runs/2026-10-14/photo-textured/grok) |

初回は写真由来の人物テクスチャを禁止した条件です。追加版では各モデル自身の初回成果物を引き継ぎ、写真の色やテクスチャを本物の立体メッシュへ使うことを許可しました。

| 共通指示 | 内容 |
| --- | --- |
| [初回の指示全文](benchmarks/cafe-selfie-v1/prompts/01-procedural.md) | 写真を参考にBlenderで制作 |
| [追加版の指示全文](benchmarks/cafe-selfie-v1/prompts/02-photo-textured.md) | 自分の初回版から写真テクスチャも使って制作 |

ハーネス・画像へのアクセス方法・処理時間・通信再試行は同一ではありません。モデル単体の一般的な優劣を証明する比較ではありません。

## 入手できるもの

通常のGitには、元写真、共通ベースメッシュ、入力情報、プロンプト、2026-10-14の8件のコードと付属素材、元のPNGに加え、KimiとTerraのシーン・GLB・検査記録も収録しています。

`.blend` と既存のWeb用 `.glb` は、**このリポジトリのRelease**に置いています。`blender-scenes.zip` をリポジトリのルートに展開すると、8件の `character.blend` と追加版の `source-character.blend` が、それぞれのコードと同じディレクトリに揃います。新しくGit LFSを設定する必要はありません。

2026-10-14の公開は、当時のファイルをコピーしたものです。公開にあたりモデルの修正、再生成、再レンダリング、再検証は行っていません。過去の生成処理の終了記録は [runs/index.json](runs/index.json) に収録しています。公開時点の新しい動作保証を表す記録ではありません。追加実行のコーディネーター検査は、それぞれの日付の `validation.json` を参照してください。

既存のWeb用GLBは初回4件と追加版Sol・Grokの計6件です。追加版Argon・OpusのGLBは新たに生成していませんが、Blenderシーンと5方向のPNGを収録しています。GLBでは髪や材質などが簡略化されるため、元のBlenderシーン・PNGも参照してください。

## ライセンス

本プロジェクトの独自コード・文書・提供された参照写真・生成成果物は [MIT](LICENSE) で公開します。MakeHumanのベースメッシュは元のCC0条件を保持します。詳細は [ASSET_LICENSES.md](ASSET_LICENSES.md) を参照してください。

