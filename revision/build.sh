#!/bin/zsh
# Build the clean revised PDF and the change-highlighted (latexdiff) PDF.
# Run AFTER make_figures.py and make_tables.py have produced figures + tables.
set -e
cd ~/logd-active-learning-benchmark/paper

echo "[build] clean manuscript..."
tectonic main.tex >/tmp/tec_main.log 2>&1 && echo "[build] main.pdf OK" || { echo "[build] main FAILED"; tail -25 /tmp/tec_main.log; exit 1; }

echo "[build] change-highlighted (latexdiff)..."
# latexdiff needs the macro/table files to exist for the revised side; the
# original had none, so flatten so \input'd tables are inlined and diffable.
latexdiff --flatten --append-safecmd="input" main_original.tex main.tex > main_diff.tex 2>/tmp/latexdiff.log || {
  echo "[build] latexdiff --flatten failed, retry without flatten";
  latexdiff main_original.tex main.tex > main_diff.tex 2>/tmp/latexdiff.log; }
tectonic main_diff.tex >/tmp/tec_diff.log 2>&1 && echo "[build] main_diff.pdf OK" || { echo "[build] diff FAILED"; tail -25 /tmp/tec_diff.log; }

echo "[build] sizes:"; ls -la main.pdf main_diff.pdf 2>/dev/null | awk '{print "  ",$5,$9}'
echo "[build] BUILD_DONE"
