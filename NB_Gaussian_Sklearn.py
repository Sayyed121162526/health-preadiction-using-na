import csv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if the input file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' not found. Please ensure the file path is correct.")
        return False
    return True

# Convert 'heartdisease.txt' to 'heartdisease.csv'
input_file = 'heartdisease.txt'
output_file = 'heartdisease.csv'

if check_file_exists(input_file):
    try:
        with open(input_file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)

            with open(output_file, 'w', newline='') as out_file:
                writer = csv.writer(out_file)

                # Write headers
                writer.writerow(('age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg',
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'))
                
                # Write data rows
                writer.writerows(lines)
        
        logging.info(f"Conversion completed! Data saved to '{output_file}'.")
    except Exception as e:
        logging.error(f"Error during file conversion: {str(e)}")
else:
    logging.warning("File conversion failed due to missing input file.")
