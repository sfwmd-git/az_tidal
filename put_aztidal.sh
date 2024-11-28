#!/bin/bash

# Define server and key details
SFTP_SERVER="sftp.sfwmd.gov"
SFTP_USER="aztidal"
KEY_PATH=".ssh/aztidal.key"

# Set the date variable
date=$(date +%Y-%m-%d)

# Define the directories and file patterns
declare -A directories
directories=(
    ["SFWMD_work/predictions/virginia_key"]="${date}*.png"
    ["SFWMD_work/predictions/vaca_key"]="${date}*.png"
    ["SFWMD_work/predictions/key_west"]="${date}*.png"
    ["SFWMD_work/predictions/lake_worth"]="${date}*.png"
    ["SFWMD_work/predictions/port_everglades"]="${date}*.png"
    ["SFWMD_work/predictions/naples"]="${date}*.png"
    ["SFWMD_work/predictions/fort_myers"]="${date}*.png"
    ["SFWMD_work/predictions/naples_bay"]="${date}*.png"
    ["SFWMD_work/predictions/all_stations"]="${date}*COMBINED*.csv"
)

# Run the SFTP command for each directory and file pattern
for dir in "${!directories[@]}"; do
    files="${directories[$dir]}"
    sftp -i $KEY_PATH $SFTP_USER@$SFTP_SERVER << EOF

cd $dir
lcd $HOME/$dir
mput $files
bye
EOF
done

echo "Files put to aztidal"
