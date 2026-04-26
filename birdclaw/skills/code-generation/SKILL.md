---
name: code-generation
description: Write or modify code files surgically, one function at a time
tags: [code, script, program, write, create, function, implement, python, class, module, scraper, bot, tool]
stages: 3
---

## stage:1 plan
Call think(). List: filename, each function name, imports needed. No code yet.
next_tools: think

## stage:2 implement
Write ONE function using write_file (first function, new file) or edit_file (add to existing).
For edit_file to append: old_string = last unique line of file, new_string = that line + new function.
5-15 lines per call. One function at a time.
next_tools: write_file,edit_file

## stage:3 verify
Run: bash("python -m py_compile <file>"). Fix errors if any. Then answer().
next_tools: bash,answer
