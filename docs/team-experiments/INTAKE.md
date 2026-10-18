# Intake (for Dhameem's Claude, in RouteLLM)

Teammates push their code and their `experiments/<id>.md` entries on their own STAIR branches. We only read those
branches: never push to them, never merge them.

1. Find the entries: `git fetch new-origin`, then `git ls-tree -r --name-only new-origin/<branch> experiments/`.
   Read each one with `git show new-origin/<branch>:experiments/<id>.md`.
2. Copy each entry into `docs/team-experiments/<id>.md`, and add a row to the status table in SHEET.md with status
   `received`.
3. Check it:
   - the YAML parses and has the template's keys;
   - `code.commit` is on that branch;
   - a grep finds no emails, PII or secrets (if it does, ask the owner to fix their branch);
   - a walltime is set, and GPUs ≤ 2;
   - the outputs are in the `stair_common.schema` layout.
4. **Reconcile the code.** Diff the branch against STAIR `main`, and against the code our pipeline uses here.
   - For each `adopt` item, and anything else that stands out, note whether it's better than ours. Better means
     faster, simpler, more correct, or filling a gap in `stair_common`.
   - Put the notes in `docs/team-experiments/adopt-notes.md` as proposals for Dhameem to decide on. Adopting
     anything is a separate PR, credited to the owner.
5. On OMEGA:
   - clone the branch at the SHA into `~/team/<id>`, and run its setup on a compute node;
   - submit each command through a PBS job sheet with a walltime (follow `scripts/omega.pbs`), and record the
     commands in `docs/gpu-jobs-common.md`;
   - log the job IDs in `runs/common-jobs.txt`.

   No compute on the login node.
6. Move the status from `queued` to `running` to `done`, with the job IDs.
7. Validate the results CSV with `stair_common.schema.validate` and the history with `validate_history`, then send
   both back to the owner.
