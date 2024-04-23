import glob
import math
import re
import numpy as np
from tqdm import trange,tqdm
import scipy.io as scio
import os
import SimpleITK as sitk
import multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, d, CT):
        self.update(h, w, d, CT)
        self.pixels = set()  # preserve all voxels in this cluster  item=(h,w,d)
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, d, CT):
        self.h = h
        self.w = w
        self.d = d
        self.CT = CT

    def __str__(self):
        return "{},{},{}:{} ".format(self.h, self.w, self.d, self.CT)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    def __init__(self, filename, K, M):
        self.K = K
        self.M = M

        self.data, self.spacing, self.origin, self.direction = self.open_nii(filename, init=True)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_depth = self.data.shape[2]
        self.N = self.image_height * self.image_width * self.image_depth  # voxel num
        self.S = int(math.pow(self.N / self.K, 1 / 3))  # voxel initial size

        self.clusters = []
        self.label = {}  # dict, (h,w,d):cluster
        self.dis = np.full((self.image_height, self.image_width, self.image_depth), np.inf)
        self.iter_num = 0
        self.reload = False

    def open_nii(self, path, init=False):
        vol_itk = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(vol_itk)
        if init:
            return vol, vol_itk.GetSpacing(), vol_itk.GetOrigin(), vol_itk.GetDirection()
        else:
            return vol

    def reload_mat(self, path, iter_num):
        mat_data = scio.loadmat(path)
        mat_data = mat_data['shape']['result'][0, 0]
        X = mat_data['X'][0, 0][:, 0]
        Y = mat_data['Y'][0, 0][:, 0]
        Z = mat_data['Z'][0, 0][:, 0]
        CT = mat_data['CT'][0, 0][:, 0]
        point = mat_data['Point'][0, 0][0]
        for i in range(len(X)):
            cluster = Cluster(int(X[i]), int(Y[i]), int(Z[i]), CT[i])
            pixels = {tuple(p) for p in point[i]}
            cluster.pixels = pixels
            for pixel in pixels:
                assert type(pixel) == tuple
                self.label[pixel] = cluster
            self.clusters.append(cluster)
        self.update_cluster()  # update self.dis
        self.iter_num = iter_num
        self.reload = True

    def save_nii(self, path, vol):
        vol_itk = sitk.GetImageFromArray(vol)
        vol_itk.SetSpacing(self.spacing)
        vol_itk.SetOrigin(self.origin)
        vol_itk.SetDirection(self.direction)
        sitk.WriteImage(vol_itk, path)

    def make_cluster(self, h, w, d) -> Cluster:
        h = int(h)
        w = int(w)
        d = int(d)
        return Cluster(h, w, d, self.data[h][w][d])

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        d = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                while d < self.image_depth:
                    self.clusters.append(self.make_cluster(h, w, d))
                    d += self.S
                d = self.S / 2
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w, d):
        # (x,y)->(x+1,y+1)
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2
        if d + 1 >= self.image_depth:
            d = self.image_depth - 2

        gradient = self.data[h + 1][w + 1][d + 1] - self.data[h][w][d]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w, cluster.d)
            h, w, d = cluster.h, cluster.w, cluster.d
            for dh in range(-1, 2):
                _h = h + dh
                if _h < 0 or _h >= self.image_height:
                    continue
                for dw in range(-1, 2):
                    _w = w + dw
                    if _w < 0 or _w >= self.image_width:
                        continue
                    for dd in range(-1, 2):
                        _d = d + dd
                        if _d < 0 or _d >= self.image_depth:
                            continue
                        new_gradient = self.get_gradient(_h, _w, _d)
                        if new_gradient < cluster_gradient:
                            assert isinstance(cluster, Cluster)
                            cluster.update(_h, _w, _d, self.data[_h][_w][_d])
                            cluster_gradient = new_gradient

    def assignment(self):
        SS = math.pow(self.S, 2)
        MM = math.pow(self.M, 2)
        for cluster in self.clusters:
            min_idx = [max(0, cluster.h - 2 * self.S),
                       max(0, cluster.w - 2 * self.S),
                       max(0, cluster.d - 2 * self.S)]
            max_idx = [min(self.image_height, cluster.h + 2 * self.S),
                       min(self.image_width, cluster.w + 2 * self.S),
                       min(self.image_depth, cluster.d + 2 * self.S)]
            x = np.arange(min_idx[0] - cluster.h, max_idx[0] - cluster.h)
            y = np.arange(min_idx[1] - cluster.w, max_idx[1] - cluster.w)
            z = np.arange(min_idx[2] - cluster.d, max_idx[2] - cluster.d)
            y, x, z = np.meshgrid(np.square(y), np.square(x), np.square(z))
            topo_dis = x + y + z
            CT_dis = self.data[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]: max_idx[2]] - cluster.CT
            CT_dis = np.square(CT_dis)
            assert topo_dis.shape == CT_dis.shape
            dist = topo_dis / SS + CT_dis / MM
            old_dis = self.dis[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]
            idx = np.array(np.where(dist < old_dis)).transpose((1, 0))
            idx += np.array(min_idx)
            for (h, w, d) in idx:
                if (h, w, d) in self.label:
                    self.label[(h, w, d)].pixels.remove((h, w, d))
                self.label[(h, w, d)] = cluster
                cluster.pixels.add((h, w, d))
            self.dis[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] = np.minimum(old_dis, dist)

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = sum_d = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                sum_d += p[2]
                number += 1
            if number == 0:
                continue

            _h = sum_h / number
            _w = sum_w / number
            _d = sum_d / number
            _h = int(_h)
            _w = int(_w)
            _d = int(_d)
            assert isinstance(cluster, Cluster)
            cluster.update(_h, _w, _d, self.data[_h][_w][_d])
            # update cluster distance
            pixels = np.array(list(cluster.pixels))
            assert len(pixels.shape) == 2 and pixels.shape[1] == 3
            CT_dis = np.square(self.data[pixels[:, 0], pixels[:, 1], pixels[:, 2]] - cluster.CT)
            topo_dis = np.sum(np.square(pixels - np.array([cluster.h, cluster.w, cluster.d])), axis=-1)
            assert CT_dis.shape == topo_dis.shape
            dist = CT_dis / (self.M ** 2) + topo_dis / (self.S ** 2)
            self.dis[pixels[:, 0], pixels[:, 1], pixels[:, 2]] = dist

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][p[2]] = cluster.CT
            image_arr[cluster.h][cluster.w][cluster.d] = 0
        self.save_nii(name, image_arr)

    def save_current_label(self, name_vis, data):
        assert len(data.shape) == 3
        # label_arr = np.copy(data)
        label_arr_vis = np.copy(data)
        vis_class = 0
        for cluster in self.clusters:
            # class_count = [0, 0, 0, 0]  # bg, non cal, cal, lumen

            # for p in cluster.pixels:
            #     class_count[data[p[0], p[1], p[2]]] += 1

            # class_val = class_count.index(max(class_count))
            for p in cluster.pixels:
                label_arr_vis[p[0], p[1], p[2]] = vis_class
                # label_arr[p[0], p[1], p[2]] = class_val
            vis_class += 1

        self.save_nii(name_vis, label_arr_vis)
        # self.save_nii(name_label, label_arr)

    def iterate(self, write_path):
        # init
        if not self.reload:
            self.init_clusters()
            self.move_clusters()

        for i in range(self.iter_num, iteration):

            self.assignment()
            self.update_cluster()

            if i == iteration - 1:
                self.save_current_label(write_path,self.data)

    def save_mat_new(self, name_string):
        maxlen = len(self.clusters)
        X = np.zeros((maxlen, 1))
        Y = np.zeros((maxlen, 1))
        Z = np.zeros((maxlen, 1))
        CT = np.zeros((maxlen, 1))
        Point = []
        image_arr = np.copy(self.data)
        key_point_num = 0
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0], p[1], p[2]] = cluster.CT
            Point.append(list(cluster.pixels))
            X[key_point_num] = cluster.h
            Y[key_point_num] = cluster.w
            Z[key_point_num] = cluster.d

            key_point_num = key_point_num + 1

        for i in range(key_point_num):
            inter_X = int(X[i][0])
            inter_Y = int(Y[i][0])
            inter_Z = int(Z[i][0])
            CT[i] = image_arr[inter_X][inter_Y][inter_Z]
        result = {'X': X, 'Y': Y, 'Z': Z, 'CT': CT, 'Point': Point}
        final_result = dict(result=result)

        mat_name = root_save_path + 'result_mat/{name_string}.mat'.format(name_string=name_string)
        scio.savemat(mat_name, {'shape': final_result}, appendmat=True, do_compression=True)


def slic_process(vol_path):
    # image_name = os.path.join(nii_dir, vol_path)
    # name_string = vol_path.split('.')[0]
    # print('image_name is：', image_name)
    # print('name_string is ', name_string)
    supervoxel_path = vol_path.replace('_0000','_0002')
    if os.path.exists(supervoxel_path):
        print('{} has been created'.format(supervoxel_path.split('/')[-1]))
        return
    print('creating {}'.format(supervoxel_path.split('/')[-1]))
    p = SLICProcessor(vol_path, K, M)
    p.iterate(supervoxel_path)
    print('{} has been created'.format(supervoxel_path.split('/')[-1]))

    # p.save_mat_new(name_string)

def pre_mkdir():
    result_mat_path = root_save_path + 'result_mat/'
    if not os.path.exists(result_mat_path):
        os.makedirs(result_mat_path)
        print('result_mat_path is creating!')
    else:
        print('result_mat_path is created!')

    result1000_path = root_save_path + 'result1000/'
    if not os.path.exists(result1000_path):
        os.mkdir(result1000_path)
        print('result1000_path is creating!')
    else:
        print('result1000_path is created!')

    result1000_loop10 = root_save_path + 'result1000_loop{}/'.format(iteration-1)
    if not os.path.exists(result1000_loop10):
        os.mkdir(result1000_loop10)
        print('result1000_loop{loop}_path is creating!'.format(loop=iteration-1))
    else:
        print('result1000_loop{loop}_path is created!'.format(loop=iteration-1))

    label1000_path = root_save_path + 'label1000/'
    if not os.path.exists(label1000_path):
        os.mkdir(label1000_path)
        print('label1000_path is creating!')
    else:
        print('label1000_path is created!')

    label_vis1000_path = root_save_path + 'label_vis1000/'
    if not os.path.exists(label_vis1000_path):
        os.mkdir(label_vis1000_path)
        print('label_vis1000_path is creating!')
    else:
        print('label_vis1000_path is created!')

    label1000_loop10_path = root_save_path + 'label1000_loop{}/'.format(iteration-1)
    if not os.path.exists(label1000_loop10_path):
        os.mkdir(label1000_loop10_path)
        print('label1000_loop{loop}_path is creating!'.format(loop=iteration-1))
    else:
        print('label1000_loop{loop}_path is created!'.format(loop=iteration-1))

    label_vis1000_loop10_path = root_save_path + 'label_vis1000_loop{}/'.format(iteration-1)
    if not os.path.exists(label_vis1000_loop10_path):
        os.mkdir(label_vis1000_loop10_path)
        print('label_vis1000_loop{loop}_path is creating!'.format(loop=iteration-1))
    else:
        print('label_vis1000_loop{loop}_path is created!'.format(loop=iteration - 1))


nii_dir = '/root/autodl-tmp/slic/3d_crop_res/nii'  # ct path
seg_dir = '/root/autodl-tmp/slic/3d_crop_res/mask'  # mask path
root_save_path = '/root/autodl-tmp/slic/k8000m1/'  # supervoxel root path
iteration = 1
K = 8000
M = 16


if __name__ == '__main__':
    root_save_path = '/root/autodl-tmp/slic/k{}m{}/'.format(K,M)  # supervoxel root path
    print('root_save_path =', root_save_path)
    pre_mkdir()

    imglist = os.listdir(nii_dir)
    imglist.sort()

    print('length of imglist is', len(imglist))

    num_processes = min(100, mp.cpu_count())
    pool = mp.Pool(processes=num_processes)

    pool.map(slic_process, imglist)

    pool.close()
    pool.join()
