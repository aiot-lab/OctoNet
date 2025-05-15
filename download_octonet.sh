#!/bin/bash
# Script to automatically download, merge, and extract Octonet files
# Download chunks with resume support, show only progress bar
# Notifies user of space needed, proceeds automatically.

# Configuration
TOTAL_CHUNKS=16
# Generate the list of chunk file names
CHUNK_LIST=()
for i in {a..p}; do
  CHUNK_LIST+=("Octonet_chunk_a$i")
done
TAR_FILE="octonet.tar"
TARGET_DIR="dataset"
# --- Notify User of Space Requirements ---
# We know each chunk is ~48GB
EXPECTED_CHUNK_SIZE_GB=48
TOTAL_DOWNLOAD_SIZE_GB=$((TOTAL_CHUNKS * EXPECTED_CHUNK_SIZE_GB))

# Estimate total space needed at peak
ESTIMATED_PEAK_SPACE_TBytes=$(echo "scale=1; ($TOTAL_DOWNLOAD_SIZE_GB * 2.0) / 1024" | bc) # Rough estimate in TBytes

echo "=============================================="
echo "Octonet Dataset Download and Extraction Script"
echo "=============================================="
echo "This script will download ${TOTAL_CHUNKS} files (each approx ${EXPECTED_CHUNK_SIZE_GB}GB),"
echo "merge them into '${TAR_FILE}', and extract the contents."
echo "The total download size is approximately ${TOTAL_DOWNLOAD_SIZE_GB}GB."
echo "At its peak, this process may require approximately ${ESTIMATED_PEAK_SPACE_TBytes} TBytes of *free disk space* in the current directory."
echo "Please ensure you have enough space before proceeding."
echo "Starting process in 5 seconds..."
sleep 5 # Give user a few seconds to read and potentially stop with Ctrl+C

# --- Download Chunks ---
echo ""
echo "--- Downloading Octonet Chunks ---"
current_chunk_idx=0
for chunk in "${CHUNK_LIST[@]}"; do
  ((current_chunk_idx++))
  echo "**** Processing $chunk ($current_chunk_idx/$TOTAL_CHUNKS)... ****"

  # Use wget -c for resume, --show-progress for visual feedback.
  # Use -q to suppress 'resuming from...' message, but keep --show-progress
  wget -q -c --show-progress "https://huggingface.co/datasets/hku-aiot/OctoNet/resolve/main/$chunk"
  WGET_EXIT_STATUS=$? # Capture wget exit status immediately

  if [ $WGET_EXIT_STATUS -ne 0 ]; then
    echo "Error: Failed to download or resume $chunk (wget exit status: $WGET_EXIT_STATUS)."
    echo "Please check your network connection, Hugging Face access, and disk space."
    exit 1
  fi
  # If wget exited with 0, it means it completed the download OR found the file complete.
  echo "$chunk download/verification completed successfully."

done

echo "--- All chunks downloaded. ---"

# --- Merge Chunks ---
echo ""
echo "--- Merging chunks into ${TAR_FILE} ---"
# Before merging, remove any existing tar file to ensure a fresh start
if [ -f "$TAR_FILE" ]; then
    echo "Removing existing $TAR_FILE before merging."
    rm -f "$TAR_FILE" # Use -f to suppress errors if file doesn't exist or permissions issues
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to remove existing ${TAR_FILE}. Proceeding anyway."
    fi
fi

# Using cat with array expansion. Error output will be visible.
cat "${CHUNK_LIST[@]}" > "$TAR_FILE"
CAT_EXIT_STATUS=$? # Capture cat exit status

if [ $CAT_EXIT_STATUS -ne 0 ]; then
  echo "Error: Failed to merge chunks (cat exit status: $CAT_EXIT_STATUS)."
  # Do not remove downloaded chunks if merging failed, they might be needed for a retry.
  exit 1
fi
echo "Merging completed successfully. ${TAR_FILE} created."

# --- Clean up Downloaded Chunks (Reduce Storage Pressure) ---
echo ""
echo "--- Cleaning up individual chunk files ---"
# Remove the individual chunk files after successful merging
rm "${CHUNK_LIST[@]}"
if [ $? -ne 0 ]; then
    echo "Warning: Failed to remove some chunk files. Manual cleanup may be required."
fi
echo "Individual chunk files removed."

# --- Extract Tar File ---
echo ""
echo "--- Extracting ${TAR_FILE} ---"
# Check if tar file exists before attempting extraction
if [ ! -f "$TAR_FILE" ]; then
    echo "Error: ${TAR_FILE} not found after merging and cleanup phase. Cannot extract."
    exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Extract the tar file into the TARGET_DIR directory
tar -xf "$TAR_FILE" -C $TARGET_DIR --strip-components=1
TAR_EXIT_STATUS=$? # Capture tar exit status

if [ $TAR_EXIT_STATUS -ne 0 ]; then
  echo "Error: Failed to extract ${TAR_FILE} (tar exit status: $TAR_EXIT_STATUS)."
  # The octonet.tar file remains for potential debugging if extraction failed
  echo "The file '${TAR_FILE}' has been kept for inspection."
  exit 1
fi
echo "Extraction completed successfully."

echo ""
echo "=============================================="
echo "Script finished."
echo "The Octonet data should now be extracted in the current directory."
echo "=============================================="