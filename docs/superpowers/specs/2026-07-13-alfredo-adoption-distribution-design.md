# Alfredo Adoption and Distribution Design

**Fecha:** 2026-07-13  
**Estado:** aprobado por el usuario  
**Repositorio:** `https://github.com/AkiraTokashiki/Alfredo`

## Objetivo

Convertir Alfredo en un proyecto open source fácil de descubrir, instalar, probar,
integrar y compartir. El objetivo de 100k estrellas es una meta de distribución,
no una garantía técnica. La primera entrega debe optimizar la conversión desde
GitHub hasta una demo funcionando sin sacrificar la precisión de las afirmaciones.

El público principal tiene dos perfiles equivalentes:

1. desarrolladores Python que quieren una capa de memoria persistente para agentes;
2. usuarios de agentes que quieren una integración MCP rápida.

## Alcance aprobado

### README y primera impresión

Reestructurar el README para que la parte superior explique en diez segundos:

- qué es Alfredo;
- qué problema resuelve;
- por qué no es un RAG genérico;
- cómo instalarlo;
- cómo ejecutar la demo;
- cómo integrarlo por MCP;
- dónde ver el benchmark.

La hero section incluirá una descripción breve, instalación, enlaces a quickstart,
demo, MCP y benchmark, más badges verificables de release, PyPI, CI, Python y
licencia. Se añadirá un asset corto de demostración; el vídeo largo existente
seguirá sirviendo como material de presentación.

La comparación visual debe cubrir memoria simple frente a Alfredo: extracción,
ranking multifactor, supersession, forgetting, namespaces, trust y contexto
limitado. No se usarán métricas, badges o claims no verificados.

### Instalación y quickstart

El contrato de instalación primaria será:

```bash
pip install memory-agent
python -m memory_agent --offline quickstart
```

La primera ejecución debe funcionar sin API key, proveedor remoto ni descarga
obligatoria de un modelo transformer. Debe usar una vault temporal, demostrar
almacenamiento y recuperación entre turnos, mostrar evidencia básica y limpiar
sus datos temporales. La documentación explicará cómo persistir con `--db`.

La distribución debe publicar un artefacto instalable en PyPI con metadata,
versionado y dependencias coherentes. La instalación desde checkout seguirá
siendo válida para contributors.

### Demo y prueba de valor

El README incluirá una demostración corta que muestre:

1. una preferencia almacenada;
2. recuperación en una nueva sesión;
3. cambio de preferencia;
4. memoria antigua archivada o supersedida;
5. motivo/evidencia de la decisión.

Los comandos documentados deben poder copiarse y ejecutarse. Los assets deben
ser pequeños, legibles y no depender de una sesión interactiva no reproducible.

### Credibilidad técnica

Destacar de forma verificable:

- benchmark sintético de 25 usuarios, 5.000 memorias y 500 preguntas;
- comparación `raw-history`, `semantic-RAG` y `alfredo`;
- casos de contradicción, expiración, abstención, forget y prompt injection;
- hashes de dataset/configuración;
- IDs seleccionados y descartados;
- trust evidence y latencia p50/p95;
- suite de tests existente y su resultado en CI.

El README debe declarar que el benchmark usa datos sintéticos y no equivale a
una auditoría de privacidad o seguridad de producción.

### Integraciones y documentación

Mantener rutas claras y separadas para:

- SDK Python;
- MCP stdio;
- MCP HTTP;
- Hermes;
- Claude Desktop;
- Cursor;
- providers LLM compatibles.

Los adapters deben seguir usando la fachada pública de memoria, sin duplicar
retrieval, trust o lifecycle en la documentación o los ejemplos.

### GitHub y comunidad

Añadir o actualizar, sin inventar políticas inexistentes:

- descripción y topics del repositorio;
- `CONTRIBUTING.md`;
- `CODE_OF_CONDUCT.md`;
- `SECURITY.md`;
- `CHANGELOG.md`;
- templates de issues y pull requests;
- roadmap público;
- workflow de CI y build/release.

Los documentos deben indicar límites reales: Alfredo es un SDK local, no un SaaS
gestionado; dashboard, billing, hosting multi-tenant y storage gestionado quedan
fuera de esta entrega.

## Fuera de alcance

No se implementará en esta entrega:

- dashboard web;
- billing;
- SaaS gestionado;
- autenticación de plataforma;
- backend remoto obligatorio;
- SDK TypeScript completo;
- promesas de escala empresarial;
- afirmaciones de precisión, privacidad o seguridad absolutas.

Estas ideas pueden permanecer como roadmap posterior, claramente etiquetadas.

## Validación

La entrega se considera válida solo si se verifican todos estos escenarios:

1. instalación limpia con `pip install memory-agent`;
2. `python -m memory_agent --offline quickstart` sin API key;
3. demo básica y demo de lifecycle;
4. benchmark offline reproducible con fixtures comprobadas;
5. `pytest tests/ -q` verde;
6. `python -m build` exitoso;
7. `twine check dist/*` exitoso;
8. comandos del README probados en un entorno separado del checkout;
9. CI configurada para las versiones Python soportadas;
10. enlaces, assets, badges y comandos del README revisados;
11. ningún documento exige configuración no declarada;
12. el paquete publicado instala la misma API descrita en el README.

## Criterios de aceptación

- Un visitante puede entender la propuesta diferencial en menos de un minuto.
- Un usuario puede instalar y ejecutar una demostración local con dos comandos.
- Un usuario MCP tiene una receta copiable y explícita.
- El benchmark es visible, reproducible y honestamente delimitado.
- La instalación publicada y la instalación desde checkout no divergen en el
  quickstart documentado.
- La documentación no contiene claims falsos, placeholders o badges rotos.
- La suite existente sigue pasando después de los cambios.
- El resultado mejora la adopción sin convertir el SDK en una plataforma fuera de
  alcance.
