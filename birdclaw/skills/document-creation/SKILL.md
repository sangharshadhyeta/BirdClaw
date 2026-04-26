---
name: document-creation
description: Create documents section by section, not all at once
tags: [document, doc, docx, report, proposal, plan, spec, write, create, draft]
stages: 3
---

## stage:1 plan
List: filename, each section heading, one sentence per section. No content yet.
next_tools: think

## stage:2 write
Write ONE section. First section: write_file with title + section. Next sections: edit_file to append.
next_tools: write_file,edit_file

## stage:3 finalize
If DOCX requested: bash("pandoc input.md -o output.docx"). Then answer().
next_tools: bash,answer
