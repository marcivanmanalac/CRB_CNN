# Description:
# This script iterates through a range of folders named "DJI_[number]_xml" and removes the "_xml" from the name.
# The script works by searching for all folders under the specified root path that match the pattern, and then renaming each matching folder.
# A success or failure message is displayed for each folder rename operation.

# Credit:
# Script written by ChatGPT, a language model developed by OpenAI
# https://openai.com
# prompted by Marc Ivan Manalac

# Define the starting and ending folder names
# rename folders within the range
# run with " .\rename_folders.ps1 "

# Define the starting and ending folder names
$start = 199
$end = 244
$prefix = 'DJI_0'
$suffix = '_xml'
$rootPath = 'C:\Users\developer\Desktop\labeled_images_dataset_Copy\East_Waiawa_Plots\East_Waiawa_Plots_XML'


# Loop through all folders from $start to $end
for ($i = $start; $i -le $end; $i++) {
    # Define the old folder name using the current loop iteration and the $prefix and $suffix
    $oldName = "$prefix$i$suffix"
    # Define the new folder name using the current loop iteration and the $prefix, without the $suffix
    $newName = "$prefix$i"
    
    # Search for all folders under $rootPath that match the $oldName pattern
    Get-ChildItem $rootPath -Recurse -Directory | Where-Object { $_.Name -like "$oldName" } | ForEach-Object { 
        # Define the new path by removing the $suffix from the current folder path
        $newPath = $_.FullName.Replace($suffix, '')
        # Attempt to rename the folder to the new path
        try {
            Rename-Item $_.FullName $newPath
            Write-Output "Successfully renamed folder $oldName to $newName"
        } catch {
            Write-Error "Failed to rename folder $oldName to $newName"
        }
    }
}
