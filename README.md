# EPUB Vocabulary Annotator

Automated tool using Python, Gemini 2.5 Flash, and GitHub Actions to scan EPUB documents, identify archaic or obscure vocabulary, and insert standard HTML footnotes for modern readers.

## Workflow Setup

1. Navigation to **Settings > Secrets and variables > Actions** in this repository.
2. Add a **New repository secret**:
   - **Name:** `GEMINI_API_KEY`
   - **Secret:** Your Google Gemini API key.

## Usage

1. Place your source `.epub` file into the `epubs/` directory.
2. Navigate to the **Actions** tab on GitHub.
3. Select **Annotate EPUB** from the left sidebar.
4. Click **Run workflow**, specify the input/output paths, and trigger the job.
5. Once complete, download the modified file from the **Artifacts** section of the workflow run.
