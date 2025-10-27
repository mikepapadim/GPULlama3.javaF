# Simple Makefile for Maven build without tests
.PHONY: build clean package validate test help

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
	@echo "  help               - Show this help message"
