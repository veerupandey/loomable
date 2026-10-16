# Security Policy

## Reporting

Report vulnerabilities privately to the maintainers via GitHub Security Advisories
on [veerupandey/loomable](https://github.com/veerupandey/loomable) or email
`rakeshpandey820@gmail.com`. Please do not open public issues for unpatched vulns.

## Trust boundaries (beta)

| Layer | Boundary |
|-------|------------|
| **Outbound requests** | Guarded HTTP tools and explicit media downloads connect to a validated public IP and check every redirect. Deployment egress controls still apply. URLs handed to external model providers follow that provider's fetch policy. |
| **Files and SQLite** | File and workspace tools reject paths that resolve outside the configured root. Read-only SQLite uses a read-only connection and a statement authorizer. Give the process access only to the files it needs and keep checkpoint directories writable only by the service. |
| **Serve (`mount_*`)** | An optional shared `api_key=` accepts Bearer or `X-API-Key`. Every holder can act on any session exposed by the mount. |
| **Cancel** | Cooperative at tool-loop boundaries; does not hard-kill in-flight provider HTTP |

Assume serve is an edge component: place it behind your gateway when exposing agents
beyond a trusted network. Enable `api_key=` for any non-local mount. Enforce
user identity and session authorization at the gateway before exposing a mount
to multiple tenants. Workflow state reads accept a `session_id` query parameter,
and state patches and approvals accept it in the request body. Requests that omit
it use the workflow's configured default session, when available.

## Supported versions

Security fixes target the current beta line (`0.2.0b*`) and later stable releases.

## Execution sandbox (beta)

`PythonTools` / `ShellTools` / `create_deep_agent(code_exec=True, shell=True)` use a
**soft** sandbox by default (`SubprocessSandbox`: child process, timeout, cwd root,
shell deny-list). This is **not** hard multi-tenant isolation.
The child runs with the service account's filesystem and network permissions.
On POSIX, timeout and cancellation terminate the child's process group.

- Prefer `sandbox_backend="docker"` or a custom `Sandbox` for untrusted code.
- Treat `run_shell` / `run_python` as privileged: deep agent adds them to
  `require_confirmation` (HITL) when enabled.
- Browser automation should use MCP (e.g. Lightpanda), not an open host shell.
