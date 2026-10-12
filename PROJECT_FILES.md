# Cozy project files

The project header draws its divider directly below the hamburger and project name, above the dataset tabs. Clicking ☰ expands New project, Open Project, Save project, and Close project. Clicking the name enables inline editing: Enter or clicking away commits, Escape cancels. Open Project launches the browser's file chooser directly; there is no intermediate upload dialog. Selected `.cozy` files are validated before replacement. One project is active at a time. New, Open, and Close offer Save, Discard, and Cancel when there are unsaved changes. Cancel preserves the current project. New and Close return to the initial blank Untitled Project.

Save project opens a dialog with the project name and a fresh snapshot download. After requesting the download, Cozy shows “Project is saved.” This confirms the download request; the browser controls its location and completion. Save before closing offers Save project and continue, which requests the download and then performs the pending New, Open, or Close action. The app does not automatically save projects to the server.

The version 1 `.cozy` format is a ZIP archive with tagged JSON metadata, Parquet tables, and binary report images. It includes all source/derived datasets, dataset names and lineage, Inspect classifications, Transform previews and accumulated drafts, dataset controls, report metadata, authored blocks, captured tables/charts, and the latest reportable output. Fitted model caches are recalculated rather than serialized. Archives never contain Python pickle or executable objects. Open rejects unknown versions, invalid registries and oversized archives (512 MB uncompressed limit).

The × beside each dataset name in the dataset navigation row opens a confirmation naming that dataset. Remove deletes it; Cancel leaves it unchanged. Derived datasets keep their independent data; lineage displays a removed-source label. Reports retain captured evidence when a source dataset is removed. Project saving includes these snapshots.

Saved projects contain the datasets and report content. Store and share the `.cozy` file as you would the original data.

The Community Edition badge uses a green background and white text. Visit counters and voting displays/controls are no longer rendered or recorded by the app entry point.
