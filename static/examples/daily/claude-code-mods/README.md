# env-guard

A tiny Claude Code mod that:

- denies `Edit` calls on `.env` files;
- puts a short note in the status line when a `Bash` command fails.

## Try it

```bash
claude --plugin-dir ./env-guard
```

Check it first, from this folder:

```bash
claude plugin validate env-guard
claude plugin test env-guard
```
