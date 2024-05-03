### File to remove brackets in wiki_auto data
### Only used for wiki_auto data . Not needed for rest 

path1 = 'data/wiki_auto_reduced_control/wiki_auto_reduced_control.test.simple'
path2 = 'data/wiki_auto_reduced_control/wiki_auto_reduced_control.valid.simple'
path3 = 'data/wiki_auto_reduced_control/wiki_auto_reduced_control.train.simple'

paths = [path1, path2, path3]

for path in paths:
    with open(path, 'r+') as f:
        lines = f.readlines()  # Read all lines into a list

        # Iterate through each line and remove square brackets
        modified_lines = [line.replace('[', '').replace(']', '') for line in lines]

        # Move file pointer to the beginning of the file
        f.seek(0)

        # Write modified lines back to the file
        f.writelines(modified_lines)

        # Truncate the remaining content (if any)
        f.truncate()
