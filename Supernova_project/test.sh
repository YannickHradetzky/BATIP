#!/bin/bash

# Log file
LOG_FILE="coarse_out.log"

# Start logging
echo "Starting coarse data computation..." > $LOG_FILE

# Loop through years from 1950 to 2010
for year in $(seq 1950 2010)
do
    # Define the output file path
    OUTPUT_FILE="Results/coarse/pvpot_${year}.nc"    
    echo "Processing year: $year" | tee -a $LOG_FILE
    
    # Call the Python script and append output to the log
    python3 -u comp_coarse_cutting_pvpot.py $year >> $LOG_FILE 2>&1
    
    # Add error checking but do not exit on error
    if [ $? -ne 0 ]; then
        echo "Error processing year $year. Continuing to next year..." | tee -a $LOG_FILE
    fi
done

echo "Coarse data computation completed for all years" | tee -a $LOG_FILE