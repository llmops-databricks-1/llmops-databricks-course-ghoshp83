# GitHub Actions Workflows

CI/CD workflows for the Arxiv Curator project.

## Workflows

### CI (`ci.yml`)

Runs pre-commit checks, pytest, and builds the wheel.

**Triggers:**
- Push to `main`, `lecture5`, or `week5_submission`
- Manual dispatch (`workflow_dispatch`)

**Steps:**
1. `uv sync --extra ci`
2. `uv run pre-commit run --all-files`
3. `uv run pytest`
4. `uv build`

### CD (`cd.yml`)

Deploys the Databricks Asset Bundle to the `dev` target using a service principal.

**Triggers:**
- Push to `main` or `week5_submission`
- Pull requests to `main`
- Manual dispatch (`workflow_dispatch`)

**Steps:**
1. Install Databricks CLI and uv
2. Configure SPN auth (via env vars + `.databrickscfg`)
3. `databricks bundle deploy --target dev --var git_sha=<sha> --var branch=<ref>`

After deploy, the `arxiv-agent-register-deploy-pipeline` job in Databricks (defined in
[`resources/register_deploy_agent.yml`](../../resources/register_deploy_agent.yml)) handles
log/register/deploy of the agent.

## Setup

### Required GitHub secrets (per environment)

Configured under **Settings → Environments → `dev`**:

- `DATABRICKS_CLIENT_ID` — service principal application ID
- `DATABRICKS_CLIENT_SECRET` — service principal secret

### Required GitHub variables (per environment)

- `DATABRICKS_HOST` — workspace URL (e.g. `https://adb-xxx.azuredatabricks.net`)

### Service principal permissions

The SPN must have permission to:
- Deploy bundles to the target workspace paths
- Create/run jobs
- Read Unity Catalog models, Vector Search endpoints, Genie spaces, SQL warehouses
- Access the `arxiv-agent-scope` secret scope

See [`notebooks/5.2_spn_permissions.py`](../../notebooks/5.2_spn_permissions.py) for granting
permissions to the SPN on serving endpoints, vector search, Genie, and warehouses.

## Usage

### Automatic deployment

```bash
git push origin week5_submission  # or main
```

### Manual deployment

1. Go to **Actions** tab
2. Select **CD**
3. Click **Run workflow**

## Troubleshooting

- **Auth failures**: verify `DATABRICKS_CLIENT_ID`/`SECRET` are set under the `dev` environment
  and the SPN has workspace access.
- **Bundle deploy failures**: run `databricks bundle validate --target dev` locally first.
- **Agent deploy failures**: check the `arxiv-agent-register-deploy-pipeline` job logs in
  the Databricks workspace.
