from common.utils import load_df_pickle
import torch
import numpy as np


def load_data(use_numpy=False):
    motion_z = torch.rand((1000, 2), dtype=torch.float, requires_grad=True)

    if use_numpy:
        tasks = np.random.randint(0, 8, size=(1000,))
        phenos = np.random.randint(0, 13, size=(1000,))
        tasks_mask = np.random.randint(0, 2, size=(1000,))
        phenos_mask = np.random.randint(0, 2, size=(1000,))
        patient_id = np.random.randint(0, 20, size=(1000,))
    else:
        tasks = torch.randint(0, 8, size=(1000,), dtype=torch.long)
        phenos = torch.randint(0, 13, size=(1000,), dtype=torch.long)
        tasks_mask = torch.randint(0, 2, size=(1000,), dtype=torch.long)
        phenos_mask = torch.randint(0, 2, size=(1000,), dtype=torch.long)
        patient_id = torch.randint(0, 20, size=(1000,), dtype=torch.long)

    return motion_z, tasks, tasks_mask, phenos, phenos_mask, patient_id


def numpy_bool_index_select(tensor_arr, mask, select_dim=0):
    idx = np.where(mask == True)[0]
    idx_tensor = torch.LongTensor(idx)
    sliced_tensor = tensor_arr.index_select(select_dim, idx_tensor)
    return sliced_tensor


class TensorAssigner:
    def __init__(self, size):
        self.size = size
        self.helper_tensor, self.finger_print_base = None, None
        self.clean()

    def assign(self, idx, arr):
        self.helper_tensor[idx, ] = arr

    def get_fingerprint(self):
        return self.finger_print_base * self.helper_tensor

    def clean(self):
        self.helper_tensor = torch.ones(size=self.size, dtype=torch.float)
        self.finger_print_base = torch.ones(size=self.size, dtype=torch.float, requires_grad=True)

class TensorAssignerDouble(TensorAssigner):
    def assign(self, idx1, idx2, arr):
        self.helper_tensor[idx1, idx2, ] = arr


if __name__ == "__main__":
    motion_z, tasks, tasks_mask, phenos, phenos_mask, patient_id = load_data(use_numpy=True)
    true_mask = (tasks_mask==1) & (phenos_mask==1)
    sliced_z = numpy_bool_index_select(tensor_arr=motion_z, mask = true_mask)
    phenos = phenos[true_mask]
    tasks = tasks[true_mask]
    patient_id = patient_id[true_mask]

    patient_assigner = TensorAssignerDouble(size=(20, 8, 2))
    tasks_assigner = TensorAssigner(size=(8, 2))

    # Calc grand means
    for i in range(8):
        average_tasks = torch.mean(numpy_bool_index_select(tensor_arr=sliced_z, mask=(tasks==i)), dim=0)
        tasks_assigner.assign(i, average_tasks)
    aver_tasks_all = tasks_assigner.get_fingerprint()

    # Calc patient's task's means
    for i in range(20):
        for j in range(8):

            patient_task_mask = (tasks == j) & (patient_id == i)
            if np.sum(patient_task_mask) > 0:
                average_patient_task = torch.mean(numpy_bool_index_select(tensor_arr=sliced_z, mask=(tasks == j)), dim=0)

            else:
                average_patient_task = aver_tasks_all[j,]

            patient_assigner.assign(i, j, average_patient_task)

    vec = patient_assigner.get_fingerprint()
    loss = torch.mean(vec)
    loss.backward()

    import pdb
    pdb.set_trace()


