import sys
import numpy as np
import pandas as pd
import os
import argparse
import json

def export_latex_txt(args):
    csv_file = pd.read_csv(args.CSV_file)
    
    main_directory = "/".join(args.CSV_file.split('/')[:-1])
    
    path = os.path.join(main_directory, 'latex')

    for folder in np.unique(csv_file[args.seperate_folders_by].values):
        folder_name = f'{args.seperate_folders_by}{folder}'
        dir = os.path.join(path, folder_name)

        if not os.path.exists(dir):
            os.makedirs(dir)

        # Split the dataframe to have only rows of the current folder
        folder_df = csv_file[csv_file[args.seperate_folders_by] == folder]

        for file in np.unique(folder_df[args.seperate_files_by].values):
            filename = f'{args.seperate_files_by}{file}.txt'
            filename = os.path.join(dir, filename)

            file_df = folder_df[folder_df[args.seperate_files_by] == file]

            txt = str()

            # Start the file with the main directory of the results in case we want to check their settings again later
            txt += f'# {main_directory}\n\n'

            # add header of the txt file to txt
            columns_list = list()
            for i, (col_key, col_dict) in enumerate(args.columns.items()):
                columns_list.append(col_key)
                txt += col_dict['col']
                if i == len(args.columns.items()) - 1:
                    txt += '\n'
                else:
                    txt += '\t'

            results_df = file_df.reindex(columns_list, axis=1)

            # Cut-off the number of digits in tau column
            results_df['tau'] = results_df['tau'].map(lambda x: '{0:.6}'.format(x))

            txt += results_df.to_csv(sep='\t', index=False, header=False)

            with open(filename, 'w') as f:
                f.write(txt)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--CSV-file', type=str, default=None, help='path to the CSV file of the results')   
    parser.add_argument('--columns', type=json.loads, default=None, help='columns of the table in the output txt file')
    parser.add_argument('--seperate-files-by', type=str, default=None, help='seperate the output files by:')
    parser.add_argument('--seperate-folders-by', type=str, default=None, help='seperate the output folders by:')
    parser.add_argument('--method-name', type=str, default=None, help='Name of the method that the results belong to')
    # parser.add_argument('--', type=str, default=None, help='')

    args = parser.parse_args()

    if args.CSV_file is None:
        args.CSV_file = ''
    
    if args.columns is None:
        args.columns = {
                        'seed'                      : {'col': 'seed', 'label': 'Random Seed'},
                        'tau'                       : {'col': 'tau', 'label': r'$\tau$'},
                        'alpha'                     : {'col': 'alpha', 'label': r'$\alpha$'},
                        'beta'                      : {'col': 'beta', 'label': r'\beta'},
                        'val_ctl_DP_avg_0_dpv'      : {'col': 'val-dpvAvg', 'label': 'DPV Avg.'},
                        'val_ctl_DP_max_0_dpv'      : {'col': 'val-dpvMAX', 'label': 'DPV Max'},
                        'val_ctl_HSIC_alpha_beta'   : {'col': 'val-hsic', 'label': 'HSIC(Z,S)'},
                        'val_ctl_KCC_alpha_beta'    : {'col': 'val-kcc', 'label': 'KCC(Z,S)'},
                        'val_tgt_utility'           : {'col': 'val-acc', 'label': 'Utility'},
                        'train_ctl_DP_avg_0_dpv'      : {'col': 'train-dpvAvg', 'label': 'DPV Avg.'},
                        'train_ctl_DP_max_0_dpv'      : {'col': 'train-dpvMAX', 'label': 'DPV Max'},
                        'train_ctl_HSIC_alpha_beta'   : {'col': 'train-hsic', 'label': 'HSIC(Z,S)'},
                        'train_ctl_KCC_alpha_beta'    : {'col': 'train-kcc', 'label': 'KCC(Z,S)'},
                        'train_tgt_utility'           : {'col': 'train-acc', 'label': 'Utility'},
                       }

    
    if args.seperate_folders_by is None:
        # args.seperate_folders_by = 'beta'
        args.seperate_folders_by = 'alpha'

    if args.seperate_files_by is None:
        # args.seperate_files_by = 'alpha'
        args.seperate_files_by = 'beta'

    if args.method_name is None:
        args.method_name = 'NoName_Method'

    
    export_latex_txt(args)