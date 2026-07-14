# Markdown export provenance

## Scope

`memory_agent.integrations.markdown_export.export_markdown` is a one-way,
read-only projection from Alfredo's SQLite `MemoryStore` into deterministic
`<memory_id>.md` files. SQLite remains the source of truth. Exported Markdown
contains records selected from one exact namespace and active lifecycle only;
it is not an import path, synchronization protocol, backup format, or promise
that a downstream editor will preserve the data.

The exporter has no Obsidian runtime dependency. Obsidian is only an example
consumer of ordinary Markdown with YAML frontmatter. The implementation uses
Python's standard library plus Alfredo's existing public `MemoryStore` reads.

## External references

The format decisions are based on the public specifications and documentation
below, consulted as format references rather than source code:

- [YAML 1.2 specification](https://yaml.org/spec/1.2.2/) — scalar quoting and
  document delimiters.
- [CommonMark specification](https://spec.commonmark.org/) — Markdown body
  syntax and the fact that body text is kept as authored data.
- [Obsidian properties documentation](https://help.obsidian.md/Editing+and+formatting/Properties)
  — interoperability context for YAML frontmatter; no Obsidian code or plugin
  files are bundled.

The exporter does not copy implementation code, templates, prose, fixtures,
or instructional text from these references.

## Licensing

Alfredo source code in this repository is distributed under the [MIT
License](../LICENSE), copyright (c) 2026 Manija. This exporter introduces no
third-party runtime dependency and bundles no third-party assets. The linked
specifications and documentation retain their respective owners' terms; the
links above are references only and do not relicense them.

## Clean-room boundary

The implementation boundary is deliberately narrow:

1. Read `MemoryRecord` values through `MemoryStore.get_memory` and
   `MemoryStore.get_all_active_memories`.
2. Filter for the requested namespace and active lifecycle.
3. Serialize Alfredo's fixed metadata keys with escaped scalar values.
4. Write UTF-8 Markdown files under the caller-provided output directory.

No external vault, plugin, database schema, or proprietary exporter was
inspected or adapted. Stored memory content is treated as untrusted data and
is copied into the body without changing the SQLite record. Frontmatter
serialization prevents body text or scalar values from adding metadata keys;
it does not make arbitrary memory content trusted or safe to execute.

## Files and generated data

The change consists only of repository-authored source and documentation:

- `src/memory_agent/integrations/markdown_export.py`
- the `export-markdown` wiring in `src/memory_agent/cli/commands.py`
- `docs/PROVENANCE.md`

No external files were copied into the repository. Files produced by running
the exporter are user-owned projections, are not checked-in source, and may
contain the original memory text and sensitive data from the selected SQLite
vault.
