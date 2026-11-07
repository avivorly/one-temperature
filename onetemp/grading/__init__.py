"""Graders that turn sampled completions into counts.jsonl: ``math500`` (the frozen grader v2.5), ``hitom``
(multiple choice, the first valid option of the completion) and ``livecodebench`` (code execution in
LiveCodeBench's own harness, vendored as ``lcb_execution``). ``completions_io`` reads completions files and writes
counts."""
