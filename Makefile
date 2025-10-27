# Simple Makefile for Maven build without tests
.PHONY: build clean package validate test release release-prepare release-perform deploy help

# Maven wrapper
MVN = ./mvnw

# Default target
all: package

# Validate environment setup
validate:
	@echo "Running environment validation..."
	@./scripts/validate-setup.sh

# Build the project (clean and package without tests)
build: clean package

# Clean the project
clean:
	$(MVN) clean

# Package the project without running tests
package:
	$(MVN) package -DskipTests

# Run tests
test:
	$(MVN) test

# Combined clean and package
package-with-clean:
	$(MVN) clean package -DskipTests

# Validate then build
safe-build: validate build

# Deploy to repository
deploy:
	$(MVN) clean deploy

# Deploy without running tests
deploy-skip-tests:
	$(MVN) clean deploy -DskipTests

# Prepare release (version bump and tag)
release-prepare:
	@echo "Preparing release..."
	$(MVN) release:prepare

# Perform release (build and deploy)
release-perform:
	@echo "Performing release..."
	$(MVN) release:perform

# Full release (prepare + perform)
release: release-prepare release-perform

# Rollback failed release
release-rollback:
	@echo "Rolling back release..."
	$(MVN) release:rollback

# Clean release artifacts
release-clean:
	@echo "Cleaning release artifacts..."
	$(MVN) release:clean

# Display help
help:
	@echo "Available targets:"
	@echo "  all                - Same as 'package' (default)"
	@echo "  validate           - Validate environment setup"
	@echo "  build              - Clean and package (without tests)"
	@echo "  clean              - Clean the project"
	@echo "  package            - Package without running tests"
	@echo "  test               - Run tests"
	@echo "  package-with-clean - Clean and package in one command"
	@echo "  safe-build         - Validate environment then build"
	@echo ""
	@echo "Deployment targets:"
	@echo "  deploy             - Deploy artifacts to repository"
	@echo "  deploy-skip-tests  - Deploy without running tests"
	@echo ""
	@echo "Release targets:"
	@echo "  release-prepare    - Prepare release (version bump, tag)"
	@echo "  release-perform    - Perform release (build and deploy)"
	@echo "  release            - Full release (prepare + perform)"
	@echo "  release-rollback   - Rollback a failed release"
	@echo "  release-clean      - Clean release artifacts"
	@echo ""
	@echo "  help               - Show this help message"
