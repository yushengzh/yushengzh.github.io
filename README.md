# yushengzh.github.io

Personal homepage of Yusheng Zhao — <https://yushengzh.github.io>

This repository holds the **built site only**. There is no build step here:
the files are served as-is by GitHub Pages from the `master` branch root.

## Contents

| Path | What it is |
| --- | --- |
| `index.html` | Home — intro, selected publications, projects, notes |
| `about.html` | Full biography |
| `publications.html` | Complete publication list |
| `assets/` | Hashed CSS/JS bundles emitted by the build |
| `cv.pdf` | CV, linked from the nav bar |
| `portrait.jpg` | Homepage portrait |
| `projects/` | Project thumbnails |
| `.nojekyll` | Tells Pages to skip Jekyll. **Do not delete.** |
| `archive/` | The previous site (2022 jemdoc), kept for reference |

## Updating the site

The source lives outside this repository — a Vite + React + Tailwind project.
Edit it there, then rebuild and copy the output over the top of this repo:

```bash
npm run build
cp -R dist/. /path/to/yushengzh.github.io/
cd /path/to/yushengzh.github.io
git add -A && git commit -m "Update site" && git push
```

`archive/` is not touched by that command.

## Notes

- Built assets in `assets/` are content-hashed, so a rebuild adds new files
  and orphans the old ones. `git add -A` picks up the deletions automatically.
- The site is a real multi-page build, not a single-page app: `about.html`
  and `publications.html` are separate documents, so deep links work without
  any rewrite rules.
