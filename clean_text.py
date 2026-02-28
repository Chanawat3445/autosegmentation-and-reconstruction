import re

file_path = "colonog_files.txt" 
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()

clean_text = re.sub(r'\.nii\.gz', '', raw_text)

output_file = "clean_colonog_files.txt"
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(clean_text)