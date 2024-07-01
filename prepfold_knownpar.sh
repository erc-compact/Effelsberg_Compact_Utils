#!/bin/bash
#SBATCH --job-name=prepfold_knownpar
#SBATCH --output=search_%A_%a.out
#SBATCH --error=search_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48  # Adjust this based on your needs
#SBATCH --mem-per-cpu=7G
#SBATCH --time=24:00:00  # Adjust this based on your needs
#SBATCH --partition=long.q  # Partition (queue) to submit to

# Define parameters
FIL_PATH="/hercules/scratch/fkareem/Ter5/filtool_output/search_5/Ter5_search5_ds1_01_inverted.fil"
FILE_PATH="/hercules/scratch/fkareem/Ter5/par_files/new_parfiles"
TMP_DIR="/tmp/fkareem/prepfold_search5/"
OUTPUT_DIR="/hercules/scratch/fkareem/Ter5/prepfold_known/search_5/searching"
TOTAL_NODES=1  # Adjust based on the total number of nodes you need

mkdir -p $OUTPUT_DIR

# Get list of files
FILES=($(ls "$FILE_PATH"))
TOTAL_FILES=${#FILES[@]}
echo "Total files to process: $TOTAL_FILES"

singularity_image="/hercules/scratch/vishnu/singularity_images/pulsar-miner_turing-sm75.sif"
run_in_singularity="singularity exec --nv -B /hercules:/hercules $singularity_image"

# Number of CPUs to use
NUM_CPUS=$SLURM_CPUS_PER_TASK
echo "Using $NUM_CPUS CPUs"

# Function to process a file
process_file() {
  local file=$1
  local dm=$2
  echo "Processing $file with DM=$dm"
  $run_in_singularity prepfold -dm $dm -par $FILE_PATH/$file -o $OUTPUT_DIR/$file $FIL_PATH -noxwin
}

# Export the function to be used by parallel
export -f process_file
export FILE_PATH
export FIL_PATH
export OUTPUT_DIR
export run_in_singularity

# Extract DMs for all files
DMS=()
for file in "${FILES[@]}"; do
  DM=$(grep '^DM ' "$FILE_PATH/$file" | awk '{print $2}')
  DMS+=("$DM")
done

# Create a job array for parallel processing
parallel_jobs=()
for ((i=0; i<TOTAL_FILES; i++)); do
  parallel_jobs+=("${FILES[$i]} ${DMS[$i]}")
done

# Run the jobs in parallel
echo "Starting parallel processing of files"
printf "%s\n" "${parallel_jobs[@]}" | xargs -n 2 -P $NUM_CPUS bash -c 'process_file "$@"' _

echo "All files have been processed"

echo "Moving output files to PFD and PNG directories"

mkdir -p $OUTPUT_DIR/PFD
mkdir -p $OUTPUT_DIR/PNG
mv $OUTPUT_DIR/*.pfd $OUTPUT_DIR/PFD/
mv $OUTPUT_DIR/*.bestprof $OUTPUT_DIR/PFD/
mv $OUTPUT_DIR/*.pfd.ps $OUTPUT_DIR/PFD/
mv $OUTPUT_DIR/*.png $OUTPUT_DIR/PNG/
mv $OUTPUT_DIR/*.pfd.polycos $OUTPUT_DIR/PFD/

# Clean up temporary directory
echo "Cleaning up temporary directory"
rm -rf $TMP_DIR

# End of script
