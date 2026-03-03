# User Study

## Dependencies

```bash
pip install numpy Pillow google-auth google-auth-oauthlib google-api-python-client
```

| Package | Used by |
|---|---|
| `numpy` | `make_grid.py`, `make_question.py` |
| `Pillow` | `make_grid.py`, `make_question.py` |
| `google-auth` | `update_form.py` |
| `google-auth-oauthlib` | `update_form.py` |
| `google-api-python-client` | `update_form.py` |

## Quickstart

### 1. Generate question images

Use `make_question.py` to create side-by-side comparison images from two folders of renders:

```bash
python make_question.py <folder1> <folder2> [--grid-size 5x3] [--tpose-size 40] [--label-size 108]
```

This produces a `<folder1>_vs_<folder2>.jpg` in the parent directory of `folder1`.

For full-batch generation from a structured results tree (`group/subgroup/method/sample`), use:

```bash
python make_questions.py <root_folder> <output_folder> [--manifest comparisons.json] [--methods Vips,MethodA,MethodB] [--sample-names sample_001,sample_014]
```

This will:
- Scan all methods/samples under `root_folder`
- Optionally restrict to a comma-separated method list via `--methods` (must include `Vips`)
- Optionally restrict to a comma-separated sample list via `--sample-names`
- Build all `Vips` vs baseline comparisons where both exist
- Generate one comparison image per pair in `output_folder`
- Write a JSON manifest (`comparisons.json` by default)

### 2. Move images into `./user_study/`

Rename/move the generated images so they follow the `question{id}.jpg` naming convention:

```bash
mv folderA_vs_folderB.jpg ./user_study/question1.jpg
mv folderC_vs_folderD.jpg ./user_study/question2.jpg
# ... and so on
```

Make sure `./user_study/tutorial.jpg` also exists (used as the first page of the form).

### 3. Update the Google Form

Run `update_form.py` to upload the images and rebuild the form:

```bash
python update_form.py
```

This will:
- Upload all `question*.jpg` (and `tutorial.jpg`) from `./user_study/` to Google Drive.
- Clear the existing form and repopulate it with one page per question image.
- Print the form URL when done.

**Prerequisites:** `credentials.json` for Google OAuth must be present in the working directory. A `token.pickle` is created automatically after the first authentication.
