# Secret alert investigation

## Findings

- The user reported a GitGuardian alert but did not have its file path or secret type.
- GitHub's native secret-scanning endpoint returned no alerts for
  `sumanth-tp/learn-ai-ml`. The latest eight remote commits had no check runs that
  identified the GitGuardian incident. These results do not rule out a separate
  GitGuardian dashboard incident.
- An offline `detect-secrets` 1.5.0 scan of tracked files, with network verification
  disabled, flagged documentation examples, test fixtures, hashes and embedded image
  data. A separate scan of additions across 65 local commits checked common provider
  key formats and credential URLs. Neither check proves that every credential is safe.
- The Secure EHR lesson contained a literal PostgreSQL password for `fde_admin`.
  Its accompanying text said the password was published in the source repository.
  This is a candidate for the alert, not a confirmed match to GitGuardian's incident.

## Changes

- Replaced the literal password in
  `docs/projects/secure-ehr-insight/01-live-implementation.md` with `CREATE USER`
  followed by the interactive `psql` command `\password fde_admin`.
- Added ignore rules for `.env` and `.env.*`, retaining the supplied `.env.example`
  and `.env.compose` configurations. The npm lockfile remains trackable.
- Verified that the removed password no longer occurs in tracked working files,
  including binary files, and that the edited EHR lesson has no remaining findings
  from the offline scanner.
- `npm run build` completed successfully after the correction.

## Still required

- Match the GitGuardian alert's file path, commit and secret type to this finding.
- Change the password on any database where it was actually used. Removing source
  text does not invalidate an existing database password.
- Commit and deploy the correction. Historical copies remain in commits
  `56b383726793`, `c1fe03609e7a` and `203b7547eab5`.
- No credentials were tested against live services, no alerts were dismissed,
  and no Git history was rewritten.
