import glob

file_lists = glob.glob('/isilon/datasets/DG_TCGA_patches/ready_to_run/splits/*.txt')
saving_folder = '/isilon/datasets/DG_TCGA_patches/ready_to_run/splits/changed_path/'
for elem in file_lists:

    with open(elem, 'r') as file:
        file_data = file.read()
    
    file_data = file_data.replace('\\','/')
    file_data = file_data.replace('.png ','.png+ ')

    with open(elem, 'w') as file:
        file.write(file_data)