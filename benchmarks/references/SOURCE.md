# Immutable benchmark reference exports

These CSV files were exported without modifying
`RAG-HPO Tests and Data Analysis copy.xlsx` (SHA-256
`b5516972c6459542ac92059c6f5d78fc1b016adce80633779b54515972d1af14`).

| File | Source sheet and range | SHA-256 |
| --- | --- | --- |
| `csc_input.csv` | `CSC Input!A1:B117` | `e2f1e91fe2f0c66b3caf23d94259262862510b1c19d12606d769aea3fc36667f` |
| `csc_manual_annotations.csv` | `CSC Manual Annotations!A1:C1790` | `fe818ca3de907962fecf55a1cc113101994f4369727f765946795dc0a2cc4f3a` |
| `gsc_input.csv` | `GSC Input!A1:C115` | `cd77a5fcd75263c196e44aeecc8aa98aeae134ca16f58ba713a23d06e92df074` |
| `gsc_manual_annotations.csv` | `GSC Manual Annotations!A1:E1013` | `ef05e38c112cca1ca1e958bd77ee96af96001ec3a3afa6d45c24e7e8b60fcea2` |

CSC input and CSC manual annotations form one corpus. GSC input and GSC manual
annotations form a separate corpus. Patient numbers are scoped to their corpus
and must never be used to join CSC to GSC.

The lab owner authorized resolving the duplicated repository Case 68. The
repository `Test_Cases.csv` Case 68 note is now synchronized exactly from
`csc_input.csv` / workbook `CSC Input!A69:B69`. Cases 67 and 68 remain distinct,
and the ten Case 68 manual annotations come from
`CSC Manual Annotations!A960:C969`. Run `python scripts/sync_csc_case_68.py` to
verify or idempotently reproduce the synchronization.
