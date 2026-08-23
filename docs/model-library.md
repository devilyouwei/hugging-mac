# Model Library

The Model Library is the control center for local model files and loaded model
instances. Listing models is read-only: it does not download, convert, or load
anything.

## Choose a variant and runtime

A model card lists its variants and runtimes. Select a variant before preparing
files. Runtime cards then show which artifacts are required and whether the
runtime is ready to load.

The default variant and runtime are convenient starting points, not permanent
choices. You can prepare more than one runtime and switch between them.

## Artifact actions

- **Download** fetches a declared prebuilt artifact.
- **Convert** creates a supported artifact locally from an installed source.
- **Remove** deletes that artifact from managed local storage.

An artifact may support both Download and Convert when the project offers a
prebuilt copy and a local conversion path. Downloading or converting again is
allowed: the new result replaces the managed files only after the operation
finishes successfully.

Some models need reusable support files. The Studio prepares those files as part
of the selected artifact workflow and includes them in runtime readiness.

## Load and unload instances

**Load instance** becomes available when every artifact needed by a runtime is
installed. A loaded instance reserves model resources and can be used by Apps,
Games, or API requests.

Unload an instance before replacing or deleting files it is using. More than one
model or runtime can be loaded when the Mac has enough unified memory, but large
language and multimodal models may require unloading another instance first.

## Storage and deletion

Hugging Mac stores managed model files outside the Python package. The total
shown in the Model Library is local disk usage, not runtime memory usage.

Removing model files does not remove the model from the catalog. You can download
or convert them again later. Deletion is limited to files managed for that model;
custom input files are not removed.

## When a model is not ready

Check the expanded runtime card:

- **Files missing** means one or more required artifacts are not installed.
- A disabled **Convert** action normally means its source must be downloaded
  first.
- A runtime can be unavailable when the operating system, architecture, or an
  optional Python dependency is not supported.
- A failed operation leaves the previous complete artifact in place whenever one
  existed.
