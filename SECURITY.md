# Security policy

Alfredo is a local-first SDK, not a hosted security service. The code and synthetic benchmark provide useful controls and evidence, but a deployment owner remains responsible for threat modeling, access control, retention, backups, and the environment in which the SQLite vault runs.

## Reporting a vulnerability or incident

Please **do not** put exploitable details, credentials, personal data, or proof-of-concept payloads in a public issue. Report a suspected security vulnerability or security incident through the repository's [GitHub Security Advisory form](https://github.com/AkiraTokashiki/Alfredo/security/advisories/new). This is the supported private maintainer channel for security reports. If the form is unavailable, do not publish exploit details; wait until a maintainer can restore the advisory path. Do not invent or rely on a public email address.

Include, when safe:

- affected version or commit and deployment mode (CLI, Python API, MCP stdio/HTTP);
- a minimal reproduction and expected versus observed behavior;
- whether a namespace boundary, prompt-injection control, or data-retention operation is involved;
- any logs with secrets, tokens, or personal data removed.

We will acknowledge receipt in the private channel, ask clarifying questions there, and coordinate disclosure after a fix or mitigation is available. No response time, remediation deadline, or availability SLA is promised by this document.

## Deployment boundaries

- **Protect the vault.** SQLite files, their journals, backups, and benchmark reports can contain memory content. Use OS permissions, filesystem encryption, backup controls, and process isolation appropriate to the data.
- **Use namespaces deliberately.** A `namespace` scopes sessions, memories, embeddings, retrieval, stats, reinforcement, supersession, and forget operations. It is an application boundary, not authentication: a process that can open the database can bypass it. For stronger tenant isolation, use separate database files and OS identities.
- **Call explicit forget.** `forget_memory` and `memory__forget` archive the selected record in the requested namespace. Archiving removes it from active retrieval; it does not guarantee physical deletion from SQLite pages, backups, logs, or downstream copies. For an erasure request, identify and delete those copies under your retention policy, then destroy the vault when appropriate.
- **Treat memory as untrusted input.** Retrieved text can contain prompt injection, instructions, secrets, or misleading claims. Keep system/developer instructions outside memory content, apply trust and confidence policy before packing context, inspect evidence and `dropped_ids`, and do not execute recalled text automatically.
- **Control optional integrations.** MCP HTTP/SSE and LLM connectors expose local data to the client or configured provider. Bind HTTP to an appropriate interface, add authentication at the deployment boundary, use TLS when crossing hosts, and limit API-key permissions. Offline mode itself makes no API-key request, but it does not secure an exposed database.
- **Check provider compatibility.** Provider and vector-dimension guards prevent incompatible comparisons; they do not detect malicious or low-quality content. Reindex or use a separate vault when changing embedding providers.

## Sensitive data limitations

The SDK is not designed to be a secret manager, password vault, regulated-record system, or automatic PII redaction service. It may persist input text, extracted memories, metadata, embeddings, and interaction summaries. Do not store credentials, private keys, payment data, health records, or other sensitive data unless your deployment has reviewed and accepted the resulting risk, retention, access, and deletion behavior. Redact before ingestion when possible.

The checked-in Alfredo Vault benchmark uses **synthetic** records. It is a reproducible behavior benchmark, not production data; the synthetic benchmark is **not a security audit or privacy audit** and cannot certify a deployment. Run an independent security and privacy review against your own threat model.

For lifecycle details, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For integration exposure and namespace examples, see [`INTEGRATION.md`](INTEGRATION.md).
