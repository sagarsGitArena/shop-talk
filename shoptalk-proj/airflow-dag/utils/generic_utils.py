
import shutil
import os

def copy_file( source_folder, destination_folder,file_name):
    
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)
    
    
    if os.path.isfile(source_file):
        try:
            # Attempt to copy the file
            shutil.copy(source_file, destination_file)
            print(f"Copied {file_name} to {destination_folder}")
        except FileNotFoundError:
            print(f"Error: {file_name} not found in the source folder.")
        except PermissionError:
            print(f"Error: Permission denied to copy {file_name}.")
        except shutil.Error as e:
            print(f"Error: An error occurred while copying {file_name}: {e}")
        except Exception as e:
            print(f"Unexpected error while copying {file_name}: {e}")