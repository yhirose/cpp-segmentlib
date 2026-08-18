# Development build: the test suite and the trainer are opt-in, so this is the
# one that can run `just test` and `just model`.
build_dir := "build"
# Consumer-shaped build: no tests, no trainer. Benchmarks use it so their
# numbers come from what a user would actually link.
bench_dir := "build-release"

model := "models/mlp/ja-ud-gsd.mod"
# The EDLA model is a research artifact, not a shipped one: it stays in the
# gitignored corpus directory so it never reaches models/, whose contents the
# release script versions and the golden fixtures pin.
ed_model := "corpus/ud-gsd/ed.mod"
corpus_dir := "corpus/ud-gsd"

# List available recipes
default:
    @just --list

# Configure and build the library, the CLI, the tests and the trainer.
# Configuring is skipped once the build directory exists, so changing an option
# means `just clean` first.
build:
    @test -d {{build_dir}} || cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE=Release \
        -DSEGMENTLIB_BUILD_TESTS=ON -DSEGMENTLIB_BUILD_TRAINING=ON
    @cmake --build {{build_dir}} --parallel

# Run the test suite
test: build
    ctest --test-dir {{build_dir}} --output-on-failure

# Run the test cases whose name matches a pattern, e.g. `just test-only dictionary`
test-only pattern: build
    {{build_dir}}/tests/segmentlib_tests -tc="*{{pattern}}*" -s

# Build with warnings as errors, as every CI job does
lint:
    cmake -S . -B {{build_dir}}-werror -DCMAKE_BUILD_TYPE=Release \
        -DSEGMENTLIB_BUILD_TRAINING=ON -DSEGMENTLIB_WARNINGS_AS_ERRORS=ON
    cmake --build {{build_dir}}-werror --parallel

# Check that a consuming project gets the library alone (what CI's consumer job runs)
consumer:
    scripts/check_consumer.sh

# Fetch everything training and evaluation need (corpora and the UniDic dictionary)
setup: corpus dict

# Fetch UD_Japanese-GSD (training, dev, test) and UD_Japanese-PUD (out-of-domain test)
corpus:
    scripts/fetch_ud_gsd_corpus.sh
    scripts/fetch_ud_pud_corpus.sh

# Fetch UniDic and convert it into the dictionary the reference model trains on
dict:
    scripts/fetch_unidic_dict.sh

# Train the reference model. Needs `just setup` first.
model: build
    {{build_dir}}/src/cli/segmenter train --backend mlp \
        --corpus {{corpus_dir}}/train.kytea.txt \
        --dev-corpus {{corpus_dir}}/dev.kytea.txt \
        --dict {{corpus_dir}}/dict_unidic.txt \
        --model-out {{model}}

# Train the same network with EDLA instead of backpropagation, on the same
# corpus, dictionary and hyperparameters as `just model` -- holding all of that
# fixed is what makes the comparison in `just eval-ed` about the learning rule.
model-ed *args: build
    {{build_dir}}/src/cli/segmenter train --backend ed \
        --corpus {{corpus_dir}}/train.kytea.txt \
        --dev-corpus {{corpus_dir}}/dev.kytea.txt \
        --dict {{corpus_dir}}/dict_unidic.txt \
        --model-out {{ed_model}} {{args}}

# Segment stdin with the reference model, e.g. `echo 日本語の文 | just predict`
predict *args: build
    @{{build_dir}}/src/cli/segmenter predict --model {{model}} {{args}}

# Boundary P/R/F1 of the reference model on both test sets
eval: build
    @for set in ud-gsd ud-pud; do \
        printf '%-8s ' "$set"; \
        python3 scripts/eval_segmentation.py --gold corpus/$set/test.kytea.txt \
            --command "{{build_dir}}/src/cli/segmenter predict --model {{model}}" \
            | grep 'P='; \
    done

# The same numbers for the EDLA model, to compare against `just eval` directly.
# Needs `just model-ed` first.
eval-ed: build
    @for set in ud-gsd ud-pud; do \
        printf '%-8s ' "$set"; \
        python3 scripts/eval_segmentation.py --gold corpus/$set/test.kytea.txt \
            --command "{{build_dir}}/src/cli/segmenter predict --model {{ed_model}}" \
            | grep 'P='; \
    done

# Inference speed and load time of the reference model, measured in-process
bench:
    @test -d {{bench_dir}} || cmake -S . -B {{bench_dir}} -DCMAKE_BUILD_TYPE=Release \
        -DSEGMENTLIB_BUILD_TESTS=OFF -DSEGMENTLIB_BUILD_TRAINING=OFF
    @cmake --build {{bench_dir}} --parallel
    {{bench_dir}}/bench/bench_segment {{model}} {{corpus_dir}}/train.raw.txt 12

# The three-way comparison against KyTea and Vaporetto (setup: bench/README.md)
bench-all:
    bench/run.sh

# Remove the build directories
clean:
    rm -rf {{build_dir}} {{build_dir}}-werror {{bench_dir}}

# Tag and push a release (dry-run; --run to execute; bumps minor if the model changed, else patch)
release *args:
    @scripts/release.sh {{args}}
