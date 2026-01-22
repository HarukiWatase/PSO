#!/usr/bin/env perl

# VS Codeで快適に使うための設定（uplatex + dvipdfmx）
$latex = 'uplatex %O -kanji=utf8 -no-guess-input-enc -synctex=1 -interaction=nonstopmode -file-line-error %S';
$bibtex = 'upbibtex %O %B';
$makeindex = 'mendex -U %O -o %D %S';
$dvipdf = 'dvipdfmx %O -o %D %S';
$pdf_mode = 3; # dvipdfmxを使う設定

# プレビュー設定 (ビルド後に自動でプレビューを開かない設定。VS Code側で制御するため)
$pvc_view_file_via_temporary = 0;
if ($^O eq 'darwin') {
    $pdf_previewer = 'open -a "Skim"';
} else {
    $pdf_previewer = 'xdg-open';
}