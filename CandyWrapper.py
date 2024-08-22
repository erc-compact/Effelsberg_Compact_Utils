import logging
import os
import argparse
import pandas as pd
import glob
import re
from astropy.time import Time

log = logging.getLogger("cands_to_csv")

class CandsProcessor:
    def __init__(self, base_dir, output_file, pics_file, meta, fil_file, logger=None):
        self.base_dir = base_dir
        self.output_file = output_file
        self.pics_file = pics_file
        self.meta = meta
        self.fil_file = fil_file
        self.data = pd.DataFrame(columns=[
            'pointing_id', 'beam_id', 'beam_name', 'source_name', 'ra', 'dec',
            'gl', 'gb', 'mjd_start', 'utc_start', 'f0_user', 'f0_opt', 'f0_opt_err',
            'f1_user', 'f1_opt', 'f1_opt_err', 'acc_user', 'acc_opt', 'acc_opt_err',
            'dm_user', 'dm_opt', 'dm_opt_err', 'sn_fft', 'sn_fold', 'pepoch',
            'maxdm_ymw16', 'dist_ymw16', 'pics_trapum_ter5', 'pics_palfa',
            'pics_meerkat_l_sband_combined_best_recall', 'pics_palfa_meerkat_l_sband_best_fscore',
            'png_path', 'metafile_path', 'filterbank_path', 'candidate_tarball_path'
        ])
        self.logger  = logger or self._create_default_logger()
        self.pics_df = self.read_pics_file()    
        
        
    
    def _create_default_logger(self):
        FORMAT = "[%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s] %(message)s"
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
        return logging.getLogger("cand_processor")
    
    def natural_sort_key(self,s):
        """Sort key that sorts strings in a human readable way"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)',s)]
        
        
    def find_subdirectories(self):
        """List all subdirectories that match the pattern "Band_*" and sort them naturally."""
        subdirs = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('Band')]
        subdirs.sort(key=self.natural_sort_key)
        print(subdirs)
        return subdirs
        
    
    def find_cand_file(self,subdir_path):
        """ Find all .cands files in the given directory"""
        pattern = os.path.join(subdir_path, '*.cands')
        return glob.glob(pattern)
    
    # def mjd_to_utc(self, mjd):
    #     """Convert Modified Julian Date (MJD) to UTC."""
    #     t = Time(mjd, format='mjd')
    #     return t.iso
    def mjd_to_utc(self, mjd):
        """Convert Modified Julian Date (MJD) to UTC in the format YYYY-MM-DDTHH:MM:SS."""
        t = Time(mjd, format='mjd')
        # Get the ISO format date string and truncate milliseconds
        return t.to_value('iso', subfmt='date_hms').replace(' ', 'T')[:19]
    
    def read_header(self, cand_file):
        log.info(f"Reading header from {cand_file}")
        header_data = {}
        try:
            with open(cand_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        line = line[1:].strip()
                        if line:
                            key, value = line.split(None, 1)
                            header_data[key] = value
                            
        except Exception as e:
            log.error(f"Error reading header from {cand_file}: {e}")
            raise
        
        # Default values in case keys are missing
        default_values = {
            'Beam': '',
            'Source_name': '',
            'RA': '',
            'DEC': '',
            'GL': '',
            'GB': '',
            'Date': '0',
            'Pepoch': '0',
            'MaxDM_YMW16': '0'
        }

        # Use header_data values or fallback to default_values
        additional_data = {
            'pointing_id': 0,  # Example value, replace with actual extraction logic
            'beam_id': 0,
            'beam_name': header_data.get('Beam', default_values['Beam']),
            'source_name': header_data.get('Source_name', default_values['Source_name']),
            'ra': header_data.get('RA', default_values['RA']),
            'dec': header_data.get('DEC', default_values['DEC']),
            'gl': header_data.get('GL', default_values['GL']),
            'gb': header_data.get('GB', default_values['GB']),
            'mjd_start': float(header_data.get('Date', default_values['Date'])),
            'utc_start': self.mjd_to_utc(float(header_data.get('Date', default_values['Date']))),
            'pepoch': float(header_data.get('Pepoch', default_values['Pepoch'])),
            'maxdm_ymw16': float(header_data.get('MaxDM_YMW16', default_values['MaxDM_YMW16']))
        }

        return additional_data
    
    def read_pics_file(self):
        """Read the pics CSV file into the Dataframe"""
        self.logger.info(f"Reading pics file: {self.pics_file}")
        try:
            pics_df = pd.read_csv(self.pics_file)
            return pics_df
        except Exception as e:
            self.logger.error(f"Error reading pics file {self.pics_file}: {e}")
            raise
    
    def get_pics_params(self, filename):
        """Retrieve pics parameters for a given filename"""
        row = self.pics_df[self.pics_df['filename'].str.contains(filename)]
        if not row.empty:
            return {
                'pics_trapum_ter5' : row['clfl2_trapum_Ter5'].values[0],
                'pics_palfa': row['clfl2_PALFA'].values[0],
                'pics_meerkat_l_sband_combined_best_recall': row['MeerKAT_L_SBAND_COMBINED_Best_Recall'].values[0],
                'pics_palfa_meerkat_l_sband_best_fscore': row['PALFA_MeerKAT_L_SBAND_Best_Fscore'].values[0]
            }
            
        return {
            'pics_trapum_ter5': 0,
            'pics_palfa': 0,
            'pics_meerkat_l_sband_combined_best_recall': 0,
            'pics_palfa_meerkat_l_sband_best_fscore': 0
        }
    
    def process_cand_file(self,cand_file, subdir):
        """Process a .cand file and append its data to the DataFrame."""
        self.logger.info(f"Processing file: {cand_file}")
        
        try:
            
            columns_to_read = [
                'dm_old','dm_new','dm_err','dist_ymw16','f0_old',
                'f0_new','f0_err','f1_old','f1_new','f1_err','acc_old',
                'acc_new','acc_err','S/N','S/N_new'
            ]
            
            # Read the relevant columns from the .cands file
            df = pd.read_csv(cand_file, skiprows=11, usecols=columns_to_read, delim_whitespace=True)
            
            # Map the read columns to the new headers
            df.rename(columns={
                'dm_old': 'dm_user', 
                'dm_new': 'dm_opt', 
                'dm_err': 'dm_opt_err',
                'f0_old': 'f0_user', 
                'f0_new': 'f0_opt', 
                'f0_err': 'f0_opt_err',
                'f1_old': 'f1_user', 
                'f1_new': 'f1_opt', 
                'f1_err': 'f1_opt_err',
                'acc_old': 'acc_user', 
                'acc_new': 'acc_opt', 
                'acc_err': 'acc_opt_err',
                'S/N': 'sn_fft', 
                'S/N_new': 'sn_fold'
            }, inplace=True)
            
            # Add missing columns with default values
            for col in self.data.columns:
                if col not in df.columns:
                    df[col] = None
            
            # Ensure all columns are in the right order
            df = df[self.data.columns]
            
            # Add ID and header data
            header_data = self.read_header(cand_file)
            full_base_dir = os.path.join("/Users/fkareem", self.base_dir)
            for idx, row in df.iterrows():
                candidate_number = idx + 1  # Candidate number (1-based index)
                pics_parameters = self.get_pics_params(f"{subdir}_{os.path.basename(cand_file).replace('.cands', f'_{candidate_number:05}.ar')}")
                # png_path = png_path = "/Users/fkareem" + os.path.join(self.base_dir, subdir, f"{os.path.basename(cand_file).replace('.cands', f'_{candidate_number:05}.png')}")
                png_path = png_path = os.path.join(subdir, f"{os.path.basename(cand_file).replace('.cands', f'_{candidate_number:05}.png')}")

                # self.logger.debug(f"Candidate {candidate_number} -> png_path: {png_path}")

                for key, value in header_data.items():
                    if key in df.columns:
                        df.at[idx, key] = value
                    #if key is not in index, move on
                for key, value in pics_parameters.items():
                    if key in df.columns:
                        df.at[idx, key] = value
                df.at[idx, 'png_path'] = png_path
                df.at[idx, 'metafile_path'] = self.meta
                df.at[idx, 'filterbank_path'] = self.fil_file
                df.at[idx, 'candidate_tarball_path'] = 'None'
                
        # Append the processed data
            self.data = pd.concat([self.data, df], ignore_index=True)
        except Exception as e:
            self.logger.error(f"Error processing file {cand_file} : {e}")
    
    
    def process_all(self):
        """Find and process all cands file and create one single csv file out of it"""
        subdirs = self.find_subdirectories()
        self.logger.info(f"Found {len(subdirs)} subdirectories to process.")
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.base_dir, subdir)
            cand_files = self.find_cand_file(subdir_path)
            # print(cand_files)
            self.logger.info(f"Found {len(cand_files)} .cand files in {subdir}")
            
            for cand_file in cand_files:
                self.process_cand_file(cand_file, subdir)
                
               
        self.data.to_csv(self.output_file, index=False)
        self.logger.info(f"Data saved to {self.output_file}")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Script for creating candidate.csv file from psrfold output")
    parser.add_argument('--base_dir', type=str, help="Path to the base directory")
    parser.add_argument('--output_file', type=str, help="Path to the output CSV file", required=True)
    parser.add_argument('--pics_file', type=str, help="Path to the pics CSV file", required=True)
    parser.add_argument('--meta', type=str, help="Path to the meta file", required=True)
    parser.add_argument('--fil_file', type=str, help="Path to the filterbank file", required=True)
    opts = parser.parse_args()
    
    processor = CandsProcessor(opts.base_dir, opts.output_file, opts.pics_file, opts.meta, opts.fil_file)
    processor.process_all()