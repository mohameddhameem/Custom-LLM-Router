# Intake (for Dhameem's Claude, in RouteLLM)

For each `<id>.md` a teammate sends back:
1. Save it as `docs/team-experiments/<id>.md`, and add a row to the status table in SHEET.md with status `received`.
2. Check it:
   - the YAML parses and has the template's keys;
   - the SHA exists on the remote;
   - a grep finds no emails, PII or secrets (if it does, reject the entry and ask the owner to fix it);
   - a walltime is set, and GPUs ≤ 2;
   - the outputs are in the `stair_common.schema` layout.
3. On OMEGA:
   - clone the repo at the SHA into `~/team/<id>`, and run its setup on a compute node;
   - submit each command through a PBS job sheet with a walltime (follow `scripts/omega.pbs`), and record the
     commands in `docs/gpu-jobs-common.md`;
   - log the job IDs in `runs/common-jobs.txt`.

   No compute on the login node.
4. Move the status from `queued` to `running` to `done`, with the job IDs.
5. Validate the results CSV with `stair_common.schema.validate` and the history with `validate_history`, then send
   both back to the owner.
