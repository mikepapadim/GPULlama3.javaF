# LangChain4j Dependency Auto-Update Automation

This document describes the automated workflows for notifying and updating the `gpu-llama3` dependency in the [langchain4j/langchain4j](https://github.com/langchain4j/langchain4j) repository when a new version is released.

## Overview

The langchain4j project uses gpu-llama3 in their `langchain4j-gpu-llama3` module. We've created two automation workflows to keep their dependency up-to-date:

1. **Notification Workflow** - Creates an issue in langchain4j about the new version
2. **PR Workflow** - Automatically creates a pull request to update the dependency

## Current Setup

### What langchain4j Uses

File: `langchain4j-gpu-llama3/pom.xml`
```xml
<dependency>
    <groupId>io.github.beehive-lab</groupId>
    <artifactId>gpu-llama3</artifactId>
    <version>0.2.2</version>
</dependency>
```

## Workflow 1: Issue Notification (Basic)

**File:** `.github/workflows/notify-langchain4j.yml`

### What It Does
- Triggers automatically when you publish a new release
- Creates an issue in langchain4j/langchain4j repository
- Notifies them about the new version with update instructions

### Triggers
- Automatic: When a release is published
- Manual: Via workflow_dispatch with custom version

### Setup Required

1. **Create a GitHub Personal Access Token (PAT)**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a descriptive name: `LangChain4j Issue Creator`
   - Select scopes:
     - ✅ `public_repo` (access public repositories)
   - Click "Generate token"
   - **Copy the token** (you won't see it again!)

2. **Add the token as a repository secret**
   - Go to your repository settings
   - Navigate to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `LANGCHAIN4J_PAT`
   - Value: Paste the token you copied
   - Click "Add secret"

3. **Update the workflow file**
   - Edit `.github/workflows/notify-langchain4j.yml`
   - Change line with `github-token: ${{ secrets.GITHUB_TOKEN }}`
   - To: `github-token: ${{ secrets.LANGCHAIN4J_PAT }}`

### Testing

Run manually to test:
```bash
# Via GitHub UI
Go to Actions → Notify LangChain4j of New Release → Run workflow
Enter version: 0.2.2
```

## Workflow 2: Automated PR Creation (Advanced)

**File:** `.github/workflows/update-langchain4j-pr.yml`

### What It Does
- Triggers automatically when you publish a new release
- Checks out the langchain4j repository
- Updates the `gpu-llama3` version in their pom.xml
- Creates a branch and commits the change
- Opens a pull request with detailed information

### Triggers
- Automatic: When a release is published
- Manual: Via workflow_dispatch with custom version

### Setup Required

1. **Create a GitHub Personal Access Token (PAT)**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a descriptive name: `LangChain4j PR Creator`
   - Select scopes:
     - ✅ `public_repo` (access public repositories)
     - ✅ `workflow` (update GitHub Actions workflows)
   - Click "Generate token"
   - **Copy the token**

2. **Add the token as a repository secret**
   - Go to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `LANGCHAIN4J_PAT`
   - Value: Paste your token
   - Click "Add secret"

3. **The workflow is ready to use!**

### Testing

Test manually before publishing a real release:
```bash
# Via GitHub UI
Go to Actions → Auto-Update LangChain4j Dependency → Run workflow
Enter version: 0.2.3
```

This will:
1. Fork/checkout langchain4j
2. Create branch: `update-gpu-llama3-0.2.3`
3. Update the dependency version
4. Push the branch
5. Create a PR

## How It Works

### On Release

When you publish a release (e.g., v0.2.3):

1. **Version Extraction**
   - Workflow extracts version from release tag
   - Removes 'v' prefix if present: `v0.2.3` → `0.2.3`

2. **Repository Access**
   - Checks out langchain4j/langchain4j
   - Creates a new branch: `update-gpu-llama3-0.2.3`

3. **File Update**
   - Opens `langchain4j-gpu-llama3/pom.xml`
   - Finds the gpu-llama3 dependency
   - Updates version to the new release version
   - Verifies the change was successful

4. **PR Creation**
   - Commits the change
   - Pushes to the new branch
   - Creates a pull request with:
     - Descriptive title
     - Change summary
     - Links to release notes
     - Testing checklist
     - Labels: `dependencies`, `enhancement`, `automated`

### Example PR

**Title:** Update gpu-llama3 dependency to 0.2.3

**Body:**
```markdown
## Description
This PR updates the `gpu-llama3` dependency to version **0.2.3**.

## Changes
- Updated `langchain4j-gpu-llama3/pom.xml`
- New version: `0.2.3`

## Release Information
- 📦 Release: https://github.com/beehive-lab/GPULlama3.java/releases/tag/v0.2.3
- 🔗 Maven Central: https://central.sonatype.com/artifact/io.github.beehive-lab/gpu-llama3/0.2.3

## Testing
- [ ] Build succeeds with new version
- [ ] Existing tests pass
- [ ] GPU-accelerated inference works correctly
```

## Choosing Which Workflow to Use

### Use Issue Notification if:
- ✅ You want langchain4j maintainers to review before updating
- ✅ You want minimal automation
- ✅ Breaking changes require discussion
- ✅ You don't want to create PRs automatically

### Use PR Creation if:
- ✅ Most updates are backward compatible
- ✅ You want full automation
- ✅ You maintain a good release cadence
- ✅ You trust the langchain4j test suite will catch issues

### Use Both if:
- ✅ You want notification AND a ready-to-review PR
- ✅ You want maximum visibility
- ⚠️ Note: This will create both an issue and a PR

## Disable a Workflow

If you want to use only one workflow:

1. Go to: Actions tab in your repository
2. Find the workflow you want to disable
3. Click the "..." menu → Disable workflow

Or delete the workflow file:
```bash
# Disable issue notifications
rm .github/workflows/notify-langchain4j.yml

# Disable PR creation
rm .github/workflows/update-langchain4j-pr.yml
```

## Security Considerations

### PAT Security
- ✅ Use a PAT with minimal required scopes
- ✅ Never commit the PAT to your repository
- ✅ Store it only in GitHub Secrets
- ✅ Consider using a bot account for automation
- ✅ Rotate tokens periodically

### Permissions
The workflows only need:
- Read access to langchain4j repository
- Write access to create issues/PRs
- No admin or sensitive permissions required

## Troubleshooting

### "Resource not accessible by integration"
**Problem:** Default `GITHUB_TOKEN` doesn't have permission to access other repositories.
**Solution:** Create and use `LANGCHAIN4J_PAT` as described above.

### PR creation fails with 404
**Problem:** Token doesn't have correct scopes.
**Solution:** Ensure PAT has `public_repo` and `workflow` scopes.

### Changes not detected
**Problem:** Version update didn't work.
**Solution:** Check that the pom.xml pattern matching is correct. The workflow uses perl regex to find and replace the version.

### Can't push to langchain4j
**Problem:** Need fork permissions.
**Solution:** The PAT should be from an account that can fork langchain4j/langchain4j and create PRs.

## Future Enhancements

Potential improvements to consider:

1. **Changelog Integration**
   - Parse CHANGELOG.md and include in PR description
   - Auto-generate migration notes

2. **Test Integration**
   - Trigger langchain4j's CI before creating PR
   - Only create PR if tests pass

3. **Version Compatibility Check**
   - Compare Java versions
   - Check TornadoVM compatibility
   - Validate against langchain4j requirements

4. **Batch Updates**
   - If multiple versions released, combine into one PR
   - Reduce notification noise

## Related Files

- `.github/workflows/notify-langchain4j.yml` - Issue notification workflow
- `.github/workflows/update-langchain4j-pr.yml` - PR creation workflow
- This documentation: `docs/LANGCHAIN4J_AUTOMATION.md`

## Support

For issues with these workflows:
1. Check GitHub Actions logs
2. Verify PAT is correctly configured
3. Test manually using workflow_dispatch
4. Open an issue in this repository

---

**Note:** These workflows interact with external repositories. Always test with manual triggers before relying on automatic release triggers.
