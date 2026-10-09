# Cozy project files

File appears between Data and Privacy and the dataset tabs. One project is active at a time. Create and Open replace it; unsaved changes require explicit discard confirmation. Cancel preserves the current project. Opening validates the archive before offering replacement.

Save project prepares a fresh snapshot. Download .cozy project saves that snapshot through the browser. Cozy marks the snapshot as saved when the download is requested; the browser controls the download location and completion. The app does not automatically save projects to the server. A new session starts with an untitled blank project.

The version 1 `.cozy` format is a ZIP archive with tagged JSON metadata, Parquet tables, and binary report images. It includes all source/derived datasets, dataset names and lineage, Inspect classifications, Transform previews and accumulated drafts, dataset controls, report metadata, authored blocks, captured tables/charts, and the latest reportable output. Fitted model caches are recalculated rather than serialized. Archives never contain Python pickle or executable objects. Open rejects unknown versions, invalid registries and oversized archives (512 MB uncompressed limit).

The × beside the active dataset's heading removes that dataset. Derived datasets keep their independent data; lineage displays a removed-source label. Reports retain captured evidence when a source dataset is removed. Project saving includes these snapshots.

Saved projects contain the datasets and report content. Store and share the `.cozy` file as you would the original data.

The Community Edition badge uses a green background and white text. Visit counters and voting displays/controls are no longer rendered or recorded by the app entry point.
