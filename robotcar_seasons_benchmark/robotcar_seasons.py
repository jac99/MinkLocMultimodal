# Functions operating on Oxford Seasons dataset

import os
import numpy as np
import csv
import mvg as mvg
import cv2
import PIL


class RobotCarSeasonsDataset:
    # Oxford RobotCar seasons dataset
    def __init__(self, dataset_root, image_ext='.jpg'):
        self.dataset_root = dataset_root
        self.image_ext = image_ext
        self.camera_name = 'rear'    # Only rear images are used in test and train splits
        self.reference_traversal_name = 'overcast-reference'
        rel_image_folder = 'images'
        self.image_folder = os.path.join(self.dataset_root, rel_image_folder)
        assert os.path.exists(self.image_folder), 'Cannot find images folder: {}'.format(self.image_folder)

        # Index images in images folder
        self.traversals = os.listdir(self.image_folder)
        self.traversals = [e for e in self.traversals if os.path.isdir(os.path.join(self.image_folder, e))]

        self.index_ts_image = {}                # index_ts_path[image timestamp] = image path
        self.index_ts_rel_image = {}            # index_ts_path[image timestamp] = relative image path
        self.index_traversal = {}               # index_traversal[traversal_name] = [ list of image timestamps ]

        for traversal in self.traversals:
            traversal_path = os.path.join(self.image_folder, traversal)
            rel_traversal_path = os.path.join(rel_image_folder, traversal)

            cameras = os.listdir(traversal_path)
            for camera in cameras:
                if camera != self.camera_name:
                    # Skip all cameras except rear camera
                    continue
                path = os.path.join(traversal_path, camera)
                rel_path = os.path.join(rel_traversal_path, camera)
                assert os.path.isdir(path), 'Not a directory: {}'.format(path)
                images = os.listdir(path)
                images = [e for e in images if os.path.splitext(e)[1] == self.image_ext]
                # Get timestamps (file name) and path to each file
                timestamps = [os.path.splitext(e)[0] for e in images]
                rel_image_paths = [os.path.join(rel_path, e) for e in images]

                assert len(timestamps) == len(rel_image_paths)

                for ts, rel_image_path in zip(timestamps, rel_image_paths):
                    # We process only images from rear camera. They should have unique timestamps
                    assert ts not in self.index_ts_image, 'ERROR: More than one image with the same timestamp found: {}'.format(ts)
                    image_path = os.path.join(self.dataset_root, rel_image_path)
                    self.index_ts_image[ts] = image_path
                    self.index_ts_rel_image[ts] = rel_image_path
                    if traversal not in self.index_traversal:
                        self.index_traversal[traversal] = []

                    self.index_traversal[traversal].append(ts)

        assert self.reference_traversal_name in self.index_traversal, 'Cannot find images from reference-traversal'
        self.camera_intrinsics, self.camera_extrinsics = self.read_camera_parameters()

        # Read reference image poses from NVM model
        self.poses = {}
        self.read_reference_poses_from_nvm()

        # Read a list of additional poses from training .txt file
        self.read_training_poses()

        # Read a list of test images (from other traversals than overcast-reference) without the traversal name
        self.test_images_ts = self.read_test_images()

        # Print some info
        print('{} images found'.format(len(self.index_ts_image)))
        for traversal in self.index_traversal:
            print('{} images in {} traversal'.format(len(self.index_traversal[traversal]), traversal))
        print('')
        print('{} known poses'.format(len(self.poses)))
        print('{} test images/ts'.format(len(self.test_images_ts)))

        # Consistency checks
        # Verify if all reference images have poses
        for ts in self.index_traversal[self.reference_traversal_name]:
            assert ts in self.poses, 'Unknown pose for reference ts: {}'.format(ts)

        # Verify if all entries from testing file have corresponding image
        for ts in self.test_images_ts:
            assert ts in self.index_ts_image, 'Cannot find image for test ts: {}'.format(ts)

        print('')

    def read_reference_poses_from_nvm(self):
        # Read poses of images in overcast-reference traversal from NVM model
        nvm_model_path = os.path.join(self.dataset_root, '3D-models', 'all-merged', 'all.nvm')
        assert os.path.exists(nvm_model_path), 'Cannot find 3D model to read reference poses: {}'.format(nvm_model_path)

        with open(nvm_model_path) as infile:
            line = get_next_line(infile)
            # Magic string for NVM model
            assert line == 'NVM_V3'
            while True:
                line = get_next_line(infile)
                if line is None or len(line) > 0:
                    break

            number_of_images = int(line)

            # Read cameras
            for i in range(number_of_images):
                line = get_next_line(infile)
                temp = line.split()
                assert len(temp) == 11
                # Strip 2 first characters from the camera name (./)
                temp2 = temp[0][2:]
                # For unknown reason cameras are linked with .png images but jpg images are provided in the dataset
                temp2, _ = os.path.splitext(temp2)
                ts = os.path.split(temp2)[1]        # Extract timestamp

                image_path = os.path.join(self.image_folder, temp2 + '.jpg')
                assert os.path.exists(image_path), 'Cannot find image: {}'.format(image_path)

                if self.camera_name not in temp2:
                    # Skip other cameras (right or left), process only rear camera
                    continue

                # Read camera pose
                q = np.array([float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5])], dtype=np.float64)
                # t must be (3, 1) vector
                t = np.array([[float(temp[6])], [float(temp[7])], [float(temp[8])]], dtype=np.float64)
                # temp[9] = radial distortion and temp[1] = 0 are discard
                # Store pose as (R, t) tuple, such that P = R(Pw - t), where Pw are point coordinates in the world
                # reference frame
                assert ts not in self.poses, 'Duplicate pose for reference image ts: {}'.format(ts)
                self.poses[ts] = mvg.Pose(mvg.q2r(q), t)

    def read_training_poses(self):
        # Read poses of training images from a text file
        path = os.path.join(self.dataset_root, 'robotcar_v2_train.txt')
        assert os.path.exists(path), 'Cannot list of training images: {}'.format(path)

        lines = [line.rstrip('\n') for line in open(path)]
        for line in lines:
            temp = line.split()
            assert len(temp) == 17

            if self.camera_name not in temp[0]:
                # Skip other cameras (right or left), process only rear camera
                print('WARNING: unexpected camera in training file: {}'.format(temp[0]))

            image_path = os.path.join(self.image_folder, temp[0])
            assert os.path.exists(image_path), 'Cannot find image: {}'.format(image_path)
            temp1 = os.path.split(temp[0])[1]   # Get filename and extension
            ts = os.path.splitext(temp1)[0]     # Get timestamp

            # Read camera pose
            T = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                for j in range(4):
                    T[i, j] = temp[1+i*4+j]
            # Convert to (R, t) form
            R, t = mvg.se3_to_rt(T)

            assert ts not in self.poses, 'Duplicate pose for ts: {}'.format(ts)
            self.poses[ts] = mvg.Pose(R, t)

    def read_test_images(self):
        # Read test images from a text file
        path = os.path.join(self.dataset_root, 'robotcar_v2_test.txt')
        assert os.path.exists(path), 'Cannot find list of test images: {}'.format(path)

        test_images_ts = []

        lines = [line.rstrip('\n') for line in open(path)]
        for line in lines:
            temp = line.split()
            assert len(temp) == 1

            if self.camera_name not in temp[0]:
                # Skip other cameras (right or left), process only rear camera
                print('WARNING: unexpected camera in test file: {}'.format(temp[0]))

            ts = os.path.splitext(temp[0])[0]
            ts = os.path.split(ts)[1]
            test_images_ts.append(ts)

        return test_images_ts

    def read_camera_parameters(self):
        # Read camera intrinsics
        intrinsics_path = os.path.join(self.dataset_root, 'intrinsics', self.camera_name + '_intrinsics.txt')
        extrinsics_path = os.path.join(self.dataset_root, 'extrinsics', self.camera_name + '_extrinsics.txt')

        assert os.path.exists(intrinsics_path), 'Cannot find camera intrinsics file: {}'.format(intrinsics_path)
        assert os.path.exists(intrinsics_path), 'Cannot find camera extrinsics file: {}'.format(extrinsics_path)

        with open(intrinsics_path) as infile:
            line = get_next_line(infile)
            temp = line.split()
            assert len(temp) == 2
            fx = temp[1]

            line = get_next_line(infile)
            temp = line.split()
            assert len(temp) == 2
            fy = temp[1]

            line = get_next_line(infile)
            temp = line.split()
            assert len(temp) == 2
            cx = temp[1]

            line = get_next_line(infile)
            temp = line.split()
            assert len(temp) == 2
            cy = temp[1]

            camera_intrinsics = mvg.CameraK(fx, fy, cx, cy)

        with open(extrinsics_path) as infile:
            csv_reader = csv.reader(infile, delimiter=',')
            rows = []
            for row in csv_reader:
                assert len(row) == 4, 'Incorrect line: {} in extrinsics file: {}'.format(row, extrinsics_path)
                rows.append(row)

            camera_extrinsics = np.array(rows, dtype=np.float)
            assert camera_extrinsics.shape == (4, 4), 'Incorrect extrinscis matrix: {}'.format(camera_extrinsics)

        return camera_intrinsics, camera_extrinsics

    def get_pose(self, traversal_name, image_id):
        if traversal_name == RobotCarSeasonsDataset.reference_traversal_name:
            camera = self.reference_images[image_id]
        else:
            camera = self.query_images[traversal_name][image_id]

        return camera.pose

    def get_camera_K(self, camera_name):
        assert camera_name in RobotCarSeasonsDataset.camera_names
        params = self.camera_intrinsics[camera_name]
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = params.fx
        K[0, 2] = params.cx
        K[1, 1] = params.fy
        K[1, 2] = params.cy
        return K

    def find_close_images(self, traversal_name, camera_name, pose, dist_thresh=5, angle_thresh=np.pi/4):
        # Returns a list of images from the given traversal and camera that are close to the given pose
        assert isinstance(pose, mvg.Pose)

        close_images = []

        if traversal_name == RobotCarSeasonsDataset.reference_traversal_name:
            sequence = self.reference_images[camera_name]
        else:
            sequence = self.query_images[traversal_name][camera_name]

        for camera in sequence:
            # Euclidean distance between camera positions
            dist = np.linalg.norm(pose.t - camera.pose.t)
            if dist <= dist_thresh:
                # Angle between rotation matrices
                angle = mvg.angle_between_rotations(pose.R, camera.pose.R)
                if abs(angle) <= angle_thresh:
                    close_images.append(camera)

        return close_images

    def find_distant_images(self, traversal_name, camera_name, pose, dist_thresh=50):
        # Returns a list of images from the given traversal and camera that are distant from the current image
        assert isinstance(pose, mvg.Pose)

        far_images = []

        if traversal_name == RobotCarSeasonsDataset.reference_traversal_name:
            sequence = self.reference_images[camera_name]
        else:
            sequence = self.query_images[traversal_name][camera_name]

        for camera in sequence:
            # Euclidean distance between camera positions
            dist = np.linalg.norm(pose.t - camera.pose.t)
            if dist > dist_thresh:
                far_images.append(camera)

        return far_images

    def find_close_reference_images(self, traversal_name, camera_name, dist_thresh=5, angle_thresh=np.pi/4):
        # Find images in the reference traversal having close images in the given traversal

        assert traversal_name is not RobotCarSeasonsDataset.reference_traversal_name

        images_with_neighbours = []

        sequence = self.reference_images[camera_name]

        for camera_ndx, camera in enumerate(sequence):
            # Find close images in the given traversal
            close_images_l = self.find_close_images(traversal_name, camera_name, camera.pose, dist_thresh=dist_thresh,
                                                    angle_thresh=angle_thresh)
            if len(close_images_l) > 0:
                #images_with_neighbours.append((camera_ndx, len(close_images_l)))
                images_with_neighbours.append(camera_ndx)

        return images_with_neighbours

    def find_close_reference_images2(self, traversals, camera_name, dist_thresh=5, angle_thresh=np.pi/4):
        # Find images in the reference traversal having close images in given traversals (for fixed camera)
        assert isinstance(traversals, list)

        images_l = []
        for traversal_name in traversals:
            if traversal_name == RobotCarSeasonsDataset.reference_traversal_name:
                continue
            images_l.extend(self.find_close_reference_images(traversal_name, camera_name,
                                                             dist_thresh=dist_thresh, angle_thresh=angle_thresh))
        # Return a list of unique elements
        return list(set(images_l))


class Image:
    # Image in Oxford RobotCar Seasons dataset - stores image path and the pose
    def __init__(self, image_path, pose):
        self.image_path = image_path
        self.pose = pose

    def read_image(self, mode='cv2'):
        assert os.path.exists(self.image_path), 'Cannot find image: {}'.format(self.image_path)
        assert mode == 'cv2' or mode == 'pil', 'mode must be cv2 or pil'
        if mode == 'cv2':
            img = cv2.imread(self.image_path)
        else:
            img = PIL.Image.open(self.image_path)

        return img


def get_next_line(file_iter):
    # Get next non-empty line from th file iterator. Strip whitespaces, including '\n' from the end.
    while True:
        line = next(file_iter, None)
        if line is not None:
            line = line.rstrip()

        return line


if __name__ == '__main__':
    dataset_root = '/media/'
    dataset = RobotCarSeasonsDataset('/media/sf_Datasets/robotcar-seasons')

