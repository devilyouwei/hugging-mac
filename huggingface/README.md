# Hugging Face organization profile

This directory owns the source for the
[Hugging Mac organization card](https://huggingface.co/hugging-mac) and its demo
page. The initial files were imported unchanged from
[`hugging-mac/README`](https://huggingface.co/spaces/hugging-mac/README/tree/main)
on 2026-11-18.

## Files

- `organization/README.md`: organization card and static Space metadata.
- `organization/index.html`: static demo page displayed by the Space.

The demo page is a dependency-free HTML/CSS grid: two columns on desktop and
one column on small screens. All videos have native playback controls. Demo
hashes link directly to the corresponding card. No JavaScript is required.
Colors are declared on the profile and cards rather than on `:root` or `body`,
because the organization profile sanitizes and scopes the embedded HTML/CSS.

Edit these files here and review the Git diff before synchronizing. Preserve the
README front matter so the Space remains configured as a static site. Organization
cards require a public Space named `README`; see the
[Hugging Face documentation](https://huggingface.co/docs/hub/organizations-cards).

Posters and videos remain in the existing Space under `posters/` and `videos/`.
They are not duplicated in this repository. The card's relative poster links
resolve in the destination Space; they will not render in a local Markdown
preview. The HTML page already uses absolute media URLs and can be opened locally
with an internet connection. Keep these remote media paths stable when editing.

## Synchronize

From the project root, authenticate with an account that can write to the Space:

```bash
uv run hf auth login
```

Upload the maintained source files in one commit:

```bash
uv run hf upload hugging-mac/README huggingface/organization . \
  --repo-type space \
  --include README.md --include index.html \
  --commit-message "Update Hugging Mac organization profile"
```

This is an explicit publication step; local edits do not upload automatically.
The command preserves existing remote posters and videos. Do not add `--delete`.
After upload, check both the organization profile and the
[demo page](https://hugging-mac-readme.static.hf.space/), including poster rendering and
video playback. If someone edits the Space directly, reconcile those changes
here before uploading again.
