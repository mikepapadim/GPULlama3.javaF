#!/bin/bash

# Script to automatically update version across all project files
# This script extracts version from pom.xml and updates it in all relevant files

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${GREEN}=== Version Update Script ===${NC}"
echo "Project root: $PROJECT_ROOT"

# Extract version from pom.xml
echo -e "\n${YELLOW}Extracting version from pom.xml...${NC}"
VERSION=$(grep -oP '<version>\K[^<]+' "$PROJECT_ROOT/pom.xml" | head -1)

if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Could not extract version from pom.xml${NC}"
    exit 1
fi

echo -e "${GREEN}Found version: $VERSION${NC}"

# Function to update a file
update_file() {
    local file=$1
    local pattern=$2
    local replacement=$3

    if [ -f "$file" ]; then
        echo -e "\n${YELLOW}Updating $file...${NC}"
        # Use perl for cross-platform compatibility
        perl -i -pe "$pattern" "$file"
        echo -e "${GREEN}✓ Updated $file${NC}"
    else
        echo -e "${RED}Warning: $file not found${NC}"
    fi
}

# Update CITATION.cff
echo -e "\n${YELLOW}Updating CITATION.cff...${NC}"
if [ -f "$PROJECT_ROOT/CITATION.cff" ]; then
    perl -i -pe "s/^version:.*$/version: $VERSION/" "$PROJECT_ROOT/CITATION.cff"
    echo -e "${GREEN}✓ Updated CITATION.cff${NC}"
fi

# Update README.md - Maven dependency version
echo -e "\n${YELLOW}Updating README.md...${NC}"
if [ -f "$PROJECT_ROOT/README.md" ]; then
    perl -i -pe "s|<version>[^<]+</version>|<version>$VERSION</version>|g" "$PROJECT_ROOT/README.md"
    echo -e "${GREEN}✓ Updated README.md${NC}"
fi

# Update scripts/all.sh - jar file paths
echo -e "\n${YELLOW}Updating scripts/all.sh...${NC}"
if [ -f "$PROJECT_ROOT/scripts/all.sh" ]; then
    perl -i -pe "s|gpu-llama3-[0-9]+\.[0-9]+\.[0-9]+\.jar|gpu-llama3-$VERSION.jar|g" "$PROJECT_ROOT/scripts/all.sh"
    echo -e "${GREEN}✓ Updated scripts/all.sh${NC}"
fi

echo -e "\n${GREEN}=== Version update complete! ===${NC}"
echo -e "All files have been updated to version ${GREEN}$VERSION${NC}"

# Summary of changes
echo -e "\n${YELLOW}Files updated:${NC}"
echo "  - CITATION.cff"
echo "  - README.md"
echo "  - scripts/all.sh"

echo -e "\n${YELLOW}To commit these changes, run:${NC}"
echo "  git add -u"
echo "  git commit -m \"chore: sync version to $VERSION\""
