
'''
Implementation for the semantickitti io

returns 
file path: /media/wanjingyi/Diskroom/js3cnet_data/sequences/00/velodyne/000000.bin
sequence: 00
id: 000000
velodyne: 
labels:
voxels:
calib.txt
poses.txt
times.txt
'''
import sys
sys.path.append('/home/wanjingyi/Documents/codes/datasets/semanticKitti/')

from torch.utils.data import Dataset
import os
import yaml
import numpy as np

from utilities.laserscan import LaserScan,SemLaserScan

# function copied from https://github.com/PRBonn/semantic-kitti-api/blob/master/visualize_voxels.py
def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed

class SemanticKitti(Dataset):
    def __init__(self, dataset_dir: str='/media/wanjingyi/Diskroom/kitti/', dataset_config_file: str='/home/wanjingyi/Documents/codes/JS3C-Net/opt/semantic-kitti.yaml', split='train', **kwargs):
        super().__init__()
        '''Load file from given dataset directory.'''

        print('[INFO] Dataset: SemanticKitti')
        self.dataset_dir = dataset_dir

        # get the config and read the file
        self.dataset_config_file = dataset_config_file

        '''
        self.dataset_config: read dataset_config_file
        train: 0, 1, 2 ...
        valid: 8
        test: 9, 10, 11 ...
        '''
        print(f'[INFO] Reading config file: {os.path.basename(dataset_config_file)}')
        try:
            self.dataset_config = yaml.safe_load(open(self.dataset_config_file, 'r'))
            print(f'[INFO] Finished.')
        except Exception as e:
            print(f'[INFO] Error.')

        '''
        self.split_sequences: get the config split list 
        train: 0, 1, 2 ...
        '''
        self.dataset_config_split = self.dataset_config['split']
        self.split = split
        self.split_sequences = self.dataset_config_split[self.split]


        '''
        self.files: get the files for the chosen split
        sequence: 00
        scan_id: 000000
        scan_velodyne: velodyne_dir/000000.bin
        scan_labels: labels_dir/000000.bin
        scan_voxels: voxels_dir/000000.bin ...
        '''
        self.files = []
        for sequence in self.split_sequences:
            sequence = format(sequence).zfill(2) # 0 --> 01
            sequence_dir = os.path.join(dataset_dir,'sequences',sequence)

            velodyne_dir = os.path.join(sequence_dir,'velodyne')
            labels_dir = os.path.join(sequence_dir,'labels')
            voxels_dir = os.path.join(sequence_dir,'voxels')

            velodyne_scans = os.listdir(velodyne_dir)
            for scan in velodyne_scans:
                scan_id = os.path.basename(os.path.splitext(scan[0])[0])
                scan_velodyne_file = os.path.join(velodyne_dir, scan_id + '.bin')
                scan_labels_file = os.path.join(labels_dir, scan_id + '.label')
                scan_voxels_bin_file = os.path.join(voxels_dir, scan_id + '.bin')
                scan_voxels_label_file = os.path.join(voxels_dir, scan_id + '.label')
                scan_voxels_invalid_file = os.path.join(voxels_dir, scan_id + '.invalid')
                scan_voxels_occluded_file = os.path.join(voxels_dir, scan_id + '.occluded')

                self.files.append({
                    'sequence': sequence,
                    'scan_id': scan_id,
                    'scan_velodyne_file': scan_velodyne_file,
                    'scan_labels_file': scan_labels_file,
                    'scan_voxels_bin_file': scan_voxels_bin_file,
                    'scan_voxels_label_file': scan_voxels_label_file,
                    'scan_voxels_invalid_file': scan_voxels_invalid_file,
                    'scan_voxels_occluded_file': scan_voxels_occluded_file
                })

        # other oprations
        self.scanner = SemLaserScan(nclasses=len(self.dataset_config['color_map']), sem_color_dict=self.dataset_config['color_map'])
        print('[INFO] Finised Initialize Dataset.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        '''Fill dictionary with available data for given index. '''

        '''
        Load raw data
        '''
        sample = self.files[idx]
        sequence = sample['sequence']
        scan_id = sample['scan_id']
        self.scanner.open_scan(sample['scan_velodyne_file']) # stored in self.scanner.points
        self.scanner.open_label(sample['scan_labels_file'])
        scan_velodyne_xyz, scan_velodyne_remission = self.scanner.xyz, self.scanner.remissions
        scan_labels = self.scanner.sem_label
        scan_voxels_bin = unpack(np.fromfile(sample['scan_voxels_bin_file'], dtype=np.uint8)).astype(np.float32)
        scan_voxels_label = np.fromfile(sample['scan_voxels_label_file'], dtype=np.uint16).astype(np.float32)
        scan_voxels_invalid = np.fromfile(sample['scan_voxels_invalid_file'], dtype=np.uint16).astype(np.float32)
        scan_voxels_occluded = np.fromfile(sample['scan_voxels_occluded_file'], dtype=np.uint16).astype(np.float32)


        '''
        other operations
        '''


        return scan_velodyne_xyz, scan_velodyne_remission, scan_labels

    
if __name__ == '__main__':
    dataset = SemanticKitti()