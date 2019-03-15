import os
import subprocess
import nibabel as nib
import numpy as np



def skull_strip_ct_isles18(ct_image, perfusion_images):
    tissue_mask = np.sum(perfusion_images, axis=0) > 0.0
    return np.multiply(ct_image, tissue_mask)

def augment_symmetric_modality(filepath_in, path_out, additional_filepaths=None, prefix='sym'):
    output_filepaths = []

    #fsl_path = '/usr/local/fsl/'
    fsl_path = '/usr/share/fsl/5.0'
    flirt_path = os.path.join(fsl_path, 'bin/flirt')
    reg_opts = '-2D -bins 256 -cost corratio -searchrx -60 60 -searchry -60 60 -searchrz -60 60 -dof 6 -interp trilinear'

    delete_command_template = 'rm {}'
    registration_command_template = flirt_path + ' -in {} -ref {} -out {} -omat {}.mat ' + reg_opts
    transform_command_template = flirt_path + ' -in {} -applyxfm -init {} -out {} -paddingsize 1.0 -interp trilinear -ref {}'

    filename_in = os.path.basename(filepath_in)

    ct_path = os.path.join(filepath_in)
    flipped_ct_path = os.path.join(path_out, 'flipped_{}'.format(filename_in))
    sym_ct_path = os.path.join(path_out, '{}_{}'.format(prefix, filename_in))
    sym_mat_path = os.path.join(path_out, 'sym')


    original_nib = nib.load(ct_path)
    original_data = original_nib.get_data()

    # Flip and save "flipped image"
    flipped_data = np.flip(original_data, axis=0)
    img = nib.Nifti1Image(flipped_data, original_nib.affine, original_nib.header)
    nib.save(img, flipped_ct_path)

    # Register flipped to original
    registration_command = registration_command_template.format(
    flipped_ct_path, ct_path, sym_ct_path, sym_mat_path)

    print('Registering flipped to original...')
    subprocess.check_output(['bash', '-c', registration_command])
    output_filepaths.append(sym_ct_path)

    # Delete flipped
    subprocess.check_output(['bash', '-c', delete_command_template.format(flipped_ct_path)])

    # Finally, register all other modalities with sym matrix
    if additional_filepaths is not None:

        for filepath_add in additional_filepaths:
            filename_add = os.path.basename(filepath_add)

            mod_path = filepath_add
            flipped_mod_path = os.path.join(path_out, 'flipped_{}'.format(filename_add))
            sym_mod_path = os.path.join(path_out, '{}_{}'.format(prefix, filename_add))

            original_nib = nib.load(mod_path)
            original_data = original_nib.get_data()

            # Flip and save "flipped image"
            flipped_data = np.flip(original_data, axis=0)
            img = nib.Nifti1Image(flipped_data, original_nib.affine, original_nib.header)
            nib.save(img, flipped_mod_path)

            print('Transforming flipped {}'.format(filename_add))
            transform_commmand = transform_command_template.format(
                flipped_mod_path, sym_mat_path + '.mat', sym_mod_path, sym_ct_path)
            subprocess.check_output(['bash', '-c', transform_commmand])
            output_filepaths.append(sym_mod_path)

            # Delete flipped
            subprocess.check_output(['bash', '-c', delete_command_template.format(flipped_mod_path)])

        # Remove matrix file finally
        subprocess.check_output(['bash', '-c', delete_command_template.format(sym_mat_path + '.mat')])

    return output_filepaths

