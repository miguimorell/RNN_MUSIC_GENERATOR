import os
import glob

def create_master_file(sequence_length,encoded_path):
    # Define the prefix for each group of files
    prefixes = ['BS', 'CK', 'CH', 'SN']
    save_path = os.path.join(encoded_path, 'Master_File')
    delimeter = '/ ' * (sequence_length)

    with open(save_path, 'w') as output:
        for prefix in prefixes:
            files = glob.glob(os.path.join(encoded_path, prefix + '_*'))
            files.sort()

            line1 = []
            #in case we want to add the velocity
            #line2 = []
            # Merge the content from files into a single line
            for file_path in files:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    line1.append(lines[0].rstrip('\n'))
                    line1.append(delimeter)
                    #in case we want to add the velocity
                    #line2.append(lines[1].rstrip('\n'))
                    #line2.append(delimeter)

            # Write the merged content to the output file
            output.write(' '.join(line1))
            output.write('\n')
            #in case we want to add the velocity
            #output.write(' '.join(line2))
            #output.write('\n')
