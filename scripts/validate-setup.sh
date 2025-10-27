#!/bin/bash
# Setup validation script for GPULlama3.java
# Validates all prerequisites before building

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0

echo -e "${BLUE}=== GPULlama3.java Setup Validation ===${NC}\n"

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" == "OK" ]; then
        echo -e "${GREEN}[✓]${NC} $message"
    elif [ "$status" == "WARN" ]; then
        echo -e "${YELLOW}[!]${NC} $message"
        ((WARNINGS++))
    elif [ "$status" == "ERROR" ]; then
        echo -e "${RED}[✗]${NC} $message"
        ((ERRORS++))
    else
        echo -e "${BLUE}[i]${NC} $message"
    fi
}

# Check Java version
echo -e "${BLUE}Checking Java installation...${NC}"
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -ge 21 ]; then
        print_status "OK" "Java version: $(java -version 2>&1 | head -n 1)"
    else
        print_status "ERROR" "Java version $JAVA_VERSION found, but Java 21+ is required"
    fi
else
    print_status "ERROR" "Java not found in PATH"
fi

# Check JAVA_HOME
if [ -n "$JAVA_HOME" ]; then
    if [ -d "$JAVA_HOME" ]; then
        print_status "OK" "JAVA_HOME: $JAVA_HOME"
    else
        print_status "ERROR" "JAVA_HOME set but directory does not exist: $JAVA_HOME"
    fi
else
    print_status "WARN" "JAVA_HOME not set (optional but recommended)"
fi

echo ""

# Check Maven
echo -e "${BLUE}Checking Maven installation...${NC}"
if [ -f "./mvnw" ]; then
    print_status "OK" "Maven wrapper found (./mvnw)"
    MVN_VERSION=$(./mvnw --version 2>&1 | head -n 1)
    print_status "INFO" "$MVN_VERSION"
else
    print_status "ERROR" "Maven wrapper (./mvnw) not found"
fi

echo ""

# Check TornadoVM
echo -e "${BLUE}Checking TornadoVM setup...${NC}"
if [ -n "$TORNADO_SDK" ]; then
    if [ -d "$TORNADO_SDK" ]; then
        print_status "OK" "TORNADO_SDK: $TORNADO_SDK"

        # Check for key TornadoVM files
        if [ -f "$TORNADO_SDK/bin/tornado" ]; then
            print_status "OK" "TornadoVM binary found"
        else
            print_status "WARN" "TornadoVM binary not found at $TORNADO_SDK/bin/tornado"
        fi
    else
        print_status "ERROR" "TORNADO_SDK set but directory does not exist: $TORNADO_SDK"
    fi
else
    print_status "WARN" "TORNADO_SDK not set (required for running, not for building)"
fi

# Check for TornadoVM submodule
if [ -d "external/tornadovm" ]; then
    print_status "OK" "TornadoVM submodule exists at external/tornadovm"

    # Check if submodule is initialized
    if [ -f "external/tornadovm/.git" ] || [ -d "external/tornadovm/.git" ]; then
        print_status "OK" "TornadoVM submodule initialized"
    else
        print_status "WARN" "TornadoVM submodule not initialized (run: git submodule update --init)"
    fi
else
    print_status "WARN" "TornadoVM submodule not found at external/tornadovm"
fi

echo ""

# Check LLAMA_ROOT
echo -e "${BLUE}Checking project configuration...${NC}"
if [ -n "$LLAMA_ROOT" ]; then
    print_status "OK" "LLAMA_ROOT: $LLAMA_ROOT"
else
    print_status "WARN" "LLAMA_ROOT not set (optional, auto-detected by scripts)"
fi

# Check for pom.xml
if [ -f "pom.xml" ]; then
    print_status "OK" "pom.xml found"
else
    print_status "ERROR" "pom.xml not found in current directory"
fi

# Check for setup scripts
if [ -f "set_paths" ]; then
    print_status "OK" "Environment setup script found (set_paths)"
else
    print_status "WARN" "set_paths script not found"
fi

echo ""

# Check for common issues
echo -e "${BLUE}Checking for common issues...${NC}"

# Check if running from project root
if [ -f "llama-tornado" ]; then
    print_status "OK" "Running from project root (llama-tornado found)"
else
    print_status "WARN" "llama-tornado runner not found - are you in the project root?"
fi

# Check git
if command -v git &> /dev/null; then
    print_status "OK" "Git is installed"
else
    print_status "WARN" "Git not found (needed for submodule management)"
fi

# Check for build artifacts
if [ -d "target" ]; then
    print_status "INFO" "Previous build artifacts found in target/"
fi

echo ""
echo -e "${BLUE}=== Validation Summary ===${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Your environment is properly configured.${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}! $WARNINGS warning(s) found. You can proceed but some features may not work.${NC}"
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) and $WARNINGS warning(s) found.${NC}"
    echo -e "${YELLOW}Please fix the errors before building.${NC}"
    exit 1
fi
