# Release Process

This document describes how to create and publish releases for GPULlama3.java.

## Prerequisites

Before creating a release, ensure you have:

1. **Git access** - Push access to the repository
2. **GPG key** - For signing artifacts (releases only)
3. **Maven credentials** - Configured in `~/.m2/settings.xml` for publishing
4. **Clean working directory** - All changes committed

## Quick Start

### Option 1: Using Makefile (Recommended)

```bash
# Full automated release
make release

# Or step by step:
make release-prepare    # Version bump + git tag
make release-perform    # Build + deploy
```

### Option 2: Using Maven directly

```bash
# Full automated release
./mvnw release:prepare release:perform

# Or step by step:
./mvnw release:prepare
./mvnw release:perform
```

## What Happens During Release

### Step 1: release:prepare

The `release:prepare` command:
1. Runs tests to verify the build
2. Removes `-SNAPSHOT` from version (e.g., `0.2.2-SNAPSHOT` → `0.2.2`)
3. Commits the release version to git with `[release] prepare release v0.2.2`
4. Creates a git tag (e.g., `v0.2.2`)
5. Bumps version to next development version (e.g., `0.2.3-SNAPSHOT`)
6. Commits the new snapshot version with `[release] prepare for next development iteration`

**Example:**
```bash
$ ./mvnw release:prepare
What is the release version for "GPU Llama3"? (io.github.beehive-lab:gpu-llama3) 0.2.2: :
What is SCM release tag or label for "GPU Llama3"? (io.github.beehive-lab:gpu-llama3) v0.2.2: :
What is the new development version for "GPU Llama3"? (io.github.beehive-lab:gpu-llama3) 0.2.3-SNAPSHOT: :
```

### Step 2: release:perform

The `release:perform` command:
1. Checks out the release tag
2. Builds the project with the `release` profile
   - Enables GPG signing
   - Generates Javadocs
   - Creates sources JAR
3. Deploys artifacts to Maven Central
4. Cleans up temporary files

## Release Profiles

The project has a `release` profile that is automatically activated during `release:perform`:

```xml
<profile>
    <id>release</id>
    <properties>
        <gpg.skip>false</gpg.skip>           <!-- Enable signing -->
        <maven.javadoc.skip>false</maven.javadoc.skip>  <!-- Generate docs -->
    </properties>
</profile>
```

## Configuration

### Maven Settings (~/.m2/settings.xml)

You need credentials configured for publishing:

```xml
<settings>
  <servers>
    <server>
      <id>central</id>
      <username>YOUR_USERNAME</username>
      <password>YOUR_PASSWORD</password>
    </server>
  </servers>
</settings>
```

### GPG Configuration

For signing artifacts, you need:

```bash
# Generate a GPG key (if you don't have one)
gpg --gen-key

# List your keys
gpg --list-keys

# Export public key to keyserver
gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID
```

Configure Maven to use your key in `~/.m2/settings.xml`:

```xml
<profiles>
  <profile>
    <id>gpg</id>
    <properties>
      <gpg.executable>gpg</gpg.executable>
      <gpg.passphrase>YOUR_PASSPHRASE</gpg.passphrase>
    </properties>
  </profile>
</profiles>

<activeProfiles>
  <activeProfile>gpg</activeProfile>
</activeProfiles>
```

## Troubleshooting

### Release Fails - How to Rollback

If `release:prepare` fails:

```bash
make release-rollback
# or
./mvnw release:rollback
```

This will:
- Remove the release tag from git
- Revert version changes in pom.xml

### Clean Up After Failed Release

```bash
make release-clean
# or
./mvnw release:clean
```

This removes temporary files created during the release process:
- `release.properties`
- `pom.xml.releaseBackup`

### Common Issues

#### Issue: "Working directory is not clean"
**Solution:** Commit all changes before releasing
```bash
git status
git add .
git commit -m "Prepare for release"
```

#### Issue: "GPG signing failed"
**Solution:** Ensure GPG key is configured and passphrase is correct
```bash
gpg --list-secret-keys
```

#### Issue: "Authentication failed for Maven Central"
**Solution:** Check credentials in `~/.m2/settings.xml`

#### Issue: "Tests failed"
**Solution:** Run tests manually first
```bash
make test
# or
./mvnw test
```

## Release Checklist

Before releasing:

- [ ] All tests passing (`make test`)
- [ ] CHANGELOG.md updated with release notes
- [ ] README.md version references updated (if any)
- [ ] All changes committed and pushed
- [ ] Working directory clean (`git status`)
- [ ] Maven Central credentials configured
- [ ] GPG key configured for signing

During release:

- [ ] Run `make release-prepare` (or `./mvnw release:prepare`)
- [ ] Verify git tags created (`git tag -l`)
- [ ] Run `make release-perform` (or `./mvnw release:perform`)
- [ ] Verify artifacts published to Maven Central

After release:

- [ ] Push tags to remote: `git push origin --tags`
- [ ] Create GitHub release from tag
- [ ] Announce release (if applicable)

## Manual Deployment (Without Release Plugin)

If you just want to deploy without version management:

```bash
# Deploy with tests
make deploy

# Deploy without tests
make deploy-skip-tests
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., `0.2.2`)
- Increment MAJOR for incompatible API changes
- Increment MINOR for backwards-compatible new features
- Increment PATCH for backwards-compatible bug fixes

Development versions have `-SNAPSHOT` suffix (e.g., `0.2.3-SNAPSHOT`)

## Release Schedule

Releases are created as needed. Typical triggers:

- Major new features
- Critical bug fixes
- Security updates
- Quarterly maintenance releases

## See Also

- [Maven Release Plugin Documentation](https://maven.apache.org/maven-release/maven-release-plugin/)
- [Semantic Versioning](https://semver.org/)
- [Maven Central Publishing](https://central.sonatype.org/publish/)
