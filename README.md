# メモ
## フォルダ構成
- input_japan：インプットフォルダ
- oupput_japan_last：アウトプットフォルダ（データサイズが大きかったためGIT連携せずGoogleDriveへコピー。実行時こちらのファイル構造を再現しないとエラーが出る。：https://drive.google.com/drive/folders/1Z9EyTl5agupObP1o3WI8Q8LnjfwCZKj_?usp=sharing）

## 学習＆結果出力ファイル
- exe_〇〇ファイル
  - DNN/NK
    - exe_dnn_〇〇：DNNモデル
    - exe_nk_〇〇；NKモデル
  - 内挿/外挿
    - exe_{dnn/nk}_inter_〇〇：内挿モデル
    - exe_{dnn/nk}_extra_〇〇：外挿モデル

＊サフィックスが_lastや_last_lastが最新

## 結果出力のみファイル
- torch__load_〇〇ファイル
  - torch.load関数でロードモデル選択可能

＊サフィックスが_lastが最新
