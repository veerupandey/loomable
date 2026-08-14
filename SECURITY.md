# Security Policy

## Reporting

Report vulnerabilities privately to the maintainers via GitHub Security Advisories
on [veerupandey/loomable](https://github.com/veerupandey/loomable) or email
`rakeshpandey820@gmail.com`. Please do not open public issues for unpatched vulns.

## Trust boundaries (beta)

| Layer | Guarantees |
|-------|------------|
| **Toolkits** | URL fetch SSRF guards, workspace path sandbox, optional Python subprocess timeout |
| **Serve (`mount_*`)** | Optional shared `api_key=` (Bearer / `X-API-Key`). **Not** full RBAC/OIDC |
| **Cancel** | Cooperative at tool-loop boundaries; does not hard-kill in-flight provider HTTP |

Assume serve is an edge component: place it behind your gateway when exposing agents
beyond a trusted network. Enable `api_key=` for any non-local mount.

## Supported versions

Security fixes target the current beta line (`0.2.0b*`) and later stable releases.

## Execution sandbox (beta)

`PythonTools` / `ShellTools` / `create_deep_agent(code_exec=True, shell=True)` use a
**soft** sandbox by default (`SubprocessSandbox`: child process, timeout, cwd root,
shell deny-list). This is **not** hard multi-tenant isolation.

- Prefer `sandbox_backend="docker"` or a custom `Sandbox` for untrusted code.
- Treat `run_shell` / `run_python` as privileged: deep agent adds them to
  `require_confirmation` (HITL) when enabled.
- Browser automation should use MCP (e.g. Lightpanda), not an open host shell.
