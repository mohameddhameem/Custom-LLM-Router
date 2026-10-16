# Team knowledge base: the STAIR repo

The team's shared repo is [`mohameddhameem/STAIR`](https://github.com/mohameddhameem/STAIR) ("STAIR — RightRoute:
STAge-level Instance Routing"). In this repo it is the git remote `new-origin`. Each member works on their own
branch; `main` holds only a README. These notes summarise what each branch is trying, so our own submission can
point to it and be compared with it. They are read from the branches, nothing is copied: follow the links for the
code and the data.

| Branch | Member | Focus | Notes |
|---|---|---|---|
| `nonLLM` | team-member-1 (`HarrytheOrange`) | Cheap non-LLM routers per pipeline stage; AgentTTS-style answer sampling with fused prefixes, on Qwen 1.5B / 7B | [nonLLM.md](nonLLM.md) |
| `selfLearnt` | team-member-2 (`Andy-TaiHL`) | No router: fine-tuned retriever / answerer ladders plus a rulebook learnt by micro-experiments; five workstreams on AgentTTS's 500 test questions | [selfLearnt.md](selfLearnt.md) |
| `selfLearnt-colab-dev` | team-member-2 | Colab development archive (notebooks 00–28, experiment log) | [selfLearnt.md](selfLearnt.md) |
| `selfLearnt-runs` | team-member-2 | Live status page of the cluster run (one commit, replaced at each update) | [selfLearnt.md](selfLearnt.md) |
| `main` | | README only | |
| this repo (`Custom-LLM-Router`) | us | System One judge (nano-jev, Strands Decider) routing small/large models; experiments 7–9 | [comparison.md](comparison.md) |

## Snapshot read

| Branch | Commit | Date |
|---|---|---|
| `main` | `4a18c1c` | 2026-10-16 |
| `nonLLM` | `b6302ce` | 2026-10-16 |
| `selfLearnt` | `dcaa60b` | 2026-10-16 |
| `selfLearnt-colab-dev` | `badb9ba` | 2026-10-16 |
| `selfLearnt-runs` | `69161f5` | 2026-10-16 |

## Refreshing

The branches are still moving. To see what changed since this snapshot:

```bash
git fetch new-origin
git log --oneline b6302ce..new-origin/nonLLM              # new commits on nonLLM
git diff --stat b6302ce new-origin/nonLLM                 # files they touched
git show new-origin/nonLLM:nonLLM/README.md               # read a file without checking the branch out
git log --oneline dcaa60b..new-origin/selfLearnt
git show new-origin/selfLearnt-runs:README.md             # the live run status
```

Then update the notes and the snapshot table. Never push to `new-origin` from here, and never merge its branches into
this repo.
