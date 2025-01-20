
import shutil
import os

def copy_file( source_folder, destination_folder,file_name):   


    source_file = os.path.join(source_folder.strip(), file_name.strip())
    #destination_file = os.path.join(destination_folder, os.path.basename(file_name))
   
    print(f'source_file:[{source_file}]')
    print(f'destination_folder:[{destination_folder}]')   
    
    
    result = ''
    if os.path.isfile(source_file):
        try:
            print(f"Attempting to copy [{source_file}] to [{destination_folder}]")
            # Copy the specific file to the destination folder
            result = shutil.copy(source_file.strip(), destination_folder.strip())
            print(f"File successfully copied to: {result}")
            
        except FileNotFoundError:
            print(f"Error: {file_name} not found in the source folder.")
        except PermissionError:
            print(f"Error: Permission denied to copy {file_name}.")
        except shutil.Error as e:
            print(f"Error: An error occurred while copying {file_name}: {e}")
        except Exception as e:
            print(f"Unexpected error while copying {file_name}: {e}")
    else:
        print(f"Error: [{source_file}] does not exist.")
    return result