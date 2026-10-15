# 見る・開く・同じ入力で作る

## 写真と画像を見る

[参照写真](../benchmarks/cafe-selfie-v1/inputs/reference.png) と [全40画像のギャラリー](../GALLERY.md) はGitHub上で閲覧できます。通常のcloneでも同じファイルを取得できます。

```sh
git clone https://github.com/sean-sunagaku/ai-girl-3d-benchmark.git
cd ai-girl-3d-benchmark
```

## Blenderシーンを開く

[Release](https://github.com/sean-sunagaku/ai-girl-3d-benchmark/releases/tag/as-published-2026-10-15) の `blender-scenes.zip` を取得し、リポジトリのルートへ展開してください。GitHub CLIを利用する場合:

```sh
gh release download as-published-2026-10-15 \
  --repo sean-sunagaku/ai-girl-3d-benchmark \
  --pattern blender-scenes.zip --dir .downloads
python3 -m zipfile -e .downloads/blender-scenes.zip .
```

各runの `character.blend` をBlenderで開きます。例えば追加版Argonは `runs/2026-10-14/photo-textured/argon/character.blend` です。追加版の `source-character.blend` は、制作時に参照した自分の初回シーンです。

同じReleaseの6個のGLBは、元から存在する簡易なWeb用出力です。追加版ArgonとOpusはBlenderシーンを利用してください。

## 同じ入力で新しく制作する

Pythonの標準ライブラリだけで、共通入力と初回プロンプトを新しい作業ディレクトリへコピーできます。

```sh
python3 scripts/prepare_case.py .work/my-model
```

そのディレクトリを使用するAIの作業場所にして、`PROMPT.md` 全文を指示として渡します。写真は `input/reference.png` にあります。課題は実際のBlenderを使って制作するものなので、利用するAIハーネスにファイルとBlenderへのアクセスを設定してください。

写真テクスチャを許可する追加版を試す場合は、自分の完成した初回ディレクトリを指定します。

```sh
python3 scripts/prepare_case.py .work/my-model-photo \
  --stage photo-textured --previous .work/my-model
```

追加版も、その作業ディレクトリの `PROMPT.md` 全文を渡します。プロンプト中のBlenderのパスは元のMacのものです。別環境では実際の実行ファイルの場所をAIに知らせ、変更点を実験記録へ残してください。新しい制作結果が過去の成果物と同一になる保証はありません。

## 公開されたコードから作り直す場合

これは必要な人が選んで行う操作です。公開時には実行していません。元のダウンロードを残すため、対象runを別の作業ディレクトリへコピーしてから、各モデルのREADMEにある手順を使ってください。

元の一般的なコマンドは以下です。Blenderの実行ファイルの場所は環境に合わせて読み替えてください。

```sh
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --threads 3 --python build.py
```

初回Opusは `previews` と `renders` の空ディレクトリを用意し、`build.py` の後に `render.py` を実行する2工程です。これは過去の記録にある手順であり、公開のための再テストはしていません。

