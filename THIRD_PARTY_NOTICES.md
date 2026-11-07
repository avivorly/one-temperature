# Third-party notices

The code of this repository is under the MIT License (`LICENSE`), except the LiveCodeBench code below, which keeps
its own license. Both licenses are MIT, so the repository as a whole can be used under the MIT terms, provided each
notice is kept.

## LiveCodeBench test harness

- **File:** `onetemp/grading/lcb_execution.py`.
- **Origin:** LiveCodeBench, <https://github.com/LiveCodeBench/LiveCodeBench>, commit
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` (2025-07-15): `lcb_runner/evaluation/testing_util.py` and parts of
  `lcb_runner/evaluation/compute_code_generation_metrics.py`.
- **Changes:** the two fixes of LiveCodeBench pull request #103 that the paper's evaluation ran with, a pinned
  multiprocessing start method, no progress bar, and a results function in place of the pass@k summary. None changes a
  verdict. The file's own header lists every change.
- **Also from LiveCodeBench, same commit and license:** `evaluation_sample()` in `onetemp/grading/livecodebench.py`
  reads a dataset row's tests the way `lcb_runner/benchmarks/code_generation.py` (`CodeGenerationProblem`) does
  (public tests, then the private tests, decoded from their compressed form), so that the harness gets the same input.
- **License:** MIT. The full notice is kept at the top of the file and reproduced here:

```
MIT License

Copyright (c) 2024 LiveCodeBench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Everything else

The rest of the code was written for the paper. The MATH-500 grader (`onetemp/grading/math500.py`) and the Hi-ToM
grader (`onetemp/grading/hitom.py`) are the paper's own graders; they contain no code from the Hendrycks MATH or
PRM800K graders or from lm-evaluation-harness.

## Data, models and benchmarks used at run time (not included)

Nothing below is copied into this repository. The prompt builders and the graders download the public files at run
time, check them by sha256, and keep them in `~/.cache/one-temperature`. Their licenses are listed in the README,
section "Data".
