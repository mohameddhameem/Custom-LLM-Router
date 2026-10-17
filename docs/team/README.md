# Team knowledge base: the STAIR repo

The team's shared repo is [`mohameddhameem/STAIR`](https://github.com/mohameddhameem/STAIR) ("STAIR — RightRoute:
STAge-level Instance Routing"). In this repo it is the git remote `new-origin`. Each member works on their own
branch; `main` holds the README plus team-member-3's `jonathan/` folder. These notes summarise what each branch is trying, so our own submission can
point to it and be compared with it. They are read from the branches, nothing is copied: follow the links for the
code and the data.

| Branch | Member | Focus | Notes |
|---|---|---|---|
| `nonLLM` | team-member-1 (`HarrytheOrange`) | Cheap non-LLM routers per pipeline stage; AgentTTS-style answer sampling with fused prefixes, on Qwen 1.5B / 7B | [nonLLM.md](nonLLM.md) |
| `selfLearnt` | team-member-2 (`Andy-TaiHL`) | No router: fine-tuned retriever / answerer ladders plus a rulebook learnt by micro-experiments; five workstreams on AgentTTS's 500 test questions | [selfLearnt.md](selfLearnt.md) |
| `selfLearnt-colab-dev` | team-member-2 | Colab development archive (notebooks 00–28, experiment log) | [selfLearnt.md](selfLearnt.md) |
| `selfLearnt-runs` | team-member-2 | Live status page of the cluster run (one commit, replaced at each update) | [selfLearnt.md](selfLearnt.md) |
| `main` (`jonathan/`) | team-member-3 (`dohyungjahn-alt`) | Stage-level routing; the team's standard lenient answer scorer (reproduces AgentTTS's 0.746) | [jonathan.md](jonathan.md) |
| this repo (`Custom-LLM-Router`) | us | System One judge (nano-jev, Strands Decider) routing small/large models; experiments 7–9 | [comparison.md](comparison.md) |

## Snapshot read

| Branch | Commit | Date |
|---|---|---|
| `main` | `e3ff28b` | 2026-10-17 |
| `nonLLM` | `da4f08f` | 2026-10-17 |
| `selfLearnt` | `3726447` | 2026-10-17 |
| `selfLearnt-colab-dev` | `0aa1743` | 2026-10-17 |
| `selfLearnt-runs` | `69161f5` | 2026-10-16 (unchanged) |

The first read was on 2026-10-16 (`main` `4a18c1c`, `nonLLM` `b6302ce`, `selfLearnt` `dcaa60b`,
`selfLearnt-colab-dev` `badb9ba`).

## Refreshing

The branches are still moving. To see what changed since this snapshot:

```bash
git fetch new-origin
git log --oneline da4f08f..new-origin/nonLLM              # new commits on nonLLM
git diff --stat da4f08f new-origin/nonLLM                 # files they touched
git show new-origin/nonLLM:nonLLM/README.md               # read a file without checking the branch out
git log --oneline 3726447..new-origin/selfLearnt
git log --oneline 0aa1743..new-origin/selfLearnt-colab-dev
git log --oneline e3ff28b..new-origin/main                # team-member-3's jonathan/ folder
git show new-origin/selfLearnt-runs:README.md             # the live run status
```

Then update the notes and the snapshot table. Never push to `new-origin` from here, and never merge its branches into
this repo.
