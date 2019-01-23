import torch as t
import cv2
from collections import Counter
from sc_mnist_data import *

t.manual_seed(time.time())
np.random.seed(int(time.time()))


class sc_mnist_model():
    def __init__(self, cnn_sizes, fc_sizes, img_dim, dx_cnt, dx_type_cnt, localization_cnt, emb_dim=2, image_depth=3,
                 device='cuda'):
        self.image_depth = image_depth
        self.device = device

        self.img_dim = img_dim
        self.dx_cnt = dx_cnt
        self.dx_type_cnt = dx_type_cnt
        self.localization_cnt = localization_cnt
        self.emb_dim = emb_dim

        self.cnn_sizes = cnn_sizes
        self.fc_sizes = fc_sizes
        return

    def create_model(self, cnn_sizes, fc_sizes, output_size):
        self.model_cnn = []
        self.model_fc = []

        self.dx_type_emb = t.tensor((self.dx_type_cnt, self.emb_dim), requires_grad=False, device=self.device)
        self.localization_emb = t.tensor((self.localization_cnt, self.emb_dim), requires_grad=False, device=self.device)

        input_size = self.image_depth
        for c in cnn_sizes:
            self.model_cnn.append(
                t.nn.Conv2d(in_channels=input_size, out_channels=c['filters'], kernel_size=c['kernel'],
                            stride=c['stride']))
            input_size = c['filters']
        [x.to(self.device) for x in self.model_cnn]

        cnn_out_dim = self._calculate_cnn_output_shape(self.img_dim, cnn_sizes) + [c['filters']]

        fc_input = np.prod(cnn_out_dim)  # + self.emb_dim * 2

        if len(list(filter(lambda x: x <= 0, cnn_out_dim))) > 0:
            raise Exception('Final conv OUTPUT dim too small, consider decreasing stride or kernel size')
        for f in fc_sizes:
            self.model_fc.append(t.nn.Linear(in_features=fc_input, out_features=f))
            fc_input = f
        [x.to(self.device) for x in self.model_fc]
        self.model_final = t.nn.Linear(in_features=fc_input, out_features=output_size, bias=False)
        self.model_final.to(self.device)

        self.model_lookup_dx_type = t.rand((self.dx_type_cnt, self.emb_dim), requires_grad=False, device=self.device)
        self.model_lookup_localization = t.rand((self.localization_cnt, self.emb_dim), requires_grad=False,
                                                device=self.device)
        self.classes = t.eye(self.dx_cnt, device=self.device)

        _h = lambda x: sum([list(y.parameters()) for y in x], [])
        self.learning_params = [
            {'params': _h(self.model_cnn) + _h(self.model_fc) + list(self.model_final.parameters())}]
        # {'params': _h(self.model_fc) + list(self.model_final.parameters())}]
        return

    def train_model(self, data, batch_size=100, iter=100, lr=1e-4, save_path=None):
        try:
            self.model_final
        except:
            self.create_model(self.cnn_sizes, self.fc_sizes, output_size=self.dx_cnt)

        data = self.balance_classes(data, 'dx')

        data = self._tensorify(data)
        last_loss = []
        self.optim = t.optim.Adam(self.learning_params, lr=lr)
        # batch_data = self.create_batch(data=data, batch_size=batch_size)
        for i in range(iter):
            print('Running iter ', i, ' of ', iter)
            try:
                batch_data = self.create_batch(data=data, batch_size=batch_size)
                pred = [[self.fprop(x), x] for x in batch_data]
                batch_loss = [self.get_loss(x[0], x[1]) for x in pred]

                l = np.median([x.data.item() for x in batch_loss])
                print('Median batch loss @ ', l)

                self.optimize_batch(batch_loss)

                if save_path:
                    def save():
                        with open(save_path, 'wb') as file_object:  # save
                            t.save(obj=self, f=file_object)
                        print('saved')

                        return

                    if t.isnan(pred[0][0]).max().data.item() == 0:
                        if last_loss == []:
                            save()
                            last_loss = l

                        elif l < last_loss:
                            save()
                            last_loss = l
            except Exception as e:
                print('Retrying loop ', str(e))
        return

    def train_model_single_label(self, data, label, batch_size=100, iter=100, lr=1e-4, save_path=None):
        try:
            self.model_final
        except:
            self.create_model(self.cnn_sizes, self.fc_sizes, output_size=2)

        data = self.balance_classes(data, 'dx')
        data = self._tensorify(data)

        self.model_final = t.nn.Linear(in_features=self.model_final.in_features, out_features=2, bias=False)
        self.model_final.to(self.device)

        last_loss = []
        best_acc = []
        acc_lim = 0.692
        self.optim = t.optim.Adam(self.learning_params, lr=lr)
        # batch_data = self.create_batch(data=data, batch_size=batch_size)
        for i in range(iter):
            print('Running iter ', i, ' of ', iter)
            try:
                batch_data = self.create_batch_for_label(data=data, label=label, batch_size=batch_size)
                pred = [[self.fprop(x), x] for x in batch_data]
                batch_loss = [self.get_loss_single_label(pred=x[0], y=x[1], label=label) for x in pred]

                l = np.median([x.data.item() for x in batch_loss])
                print('Median batch loss @ ', l)

                self.optimize_batch(batch_loss)

                if save_path:
                    def save():
                        with open(save_path + '_label_' + str(label), 'wb') as file_object:  # save
                            t.save(obj=self, f=file_object)
                        print('saved')

                        return

                    if t.isnan(pred[0][0]).max().data.item() == 0:
                        if last_loss == []:
                            save()
                            last_loss = l

                        elif l < last_loss:
                            save()
                            last_loss = l
                        best_acc.append(l)
                        if len(best_acc) >= 10:
                            if np.max(best_acc) < acc_lim:
                                print('Best acc')
                                acc_lim = np.max(best_acc)
                                save()
                            best_acc.pop(0)
            except Exception as e:
                print('Retrying loop ', str(e))
        return

    def fprop(self, data):
        data_img = cv2.imread(data['image_id'])

        # data_img = data['dx'].data.item() * np.ones((450, 600, 3))

        data_dx_type = data['dx_type']
        data_sex = data['sex']
        data_age = data['age']
        data_localization = data['localization']

        # to tensors
        data_img = t.tensor(data_img, device=self.device, dtype=t.float32).transpose(0, 2).unsqueeze(0)

        out_cnn = self._use_est(self.model_cnn, data_img, act=t.relu).reshape(-1)
        data_dx_type = t.index_select(self.model_lookup_dx_type, dim=0, index=data_dx_type)
        data_localization = t.index_select(self.model_lookup_localization, dim=0, index=data_localization)

        # fc_input = t.cat([out_cnn.reshape(1, -1), data_dx_type, data_localization], dim=1)
        fc_input = out_cnn

        out_fc = self._use_est(self.model_fc, fc_input)
        final = self.model_final(out_fc).tanh().softmax(dim=0)  # (dim=1)
        return final

    def get_loss(self, pred, data):
        ground_truth = t.index_select(self.classes, dim=0, index=data['dx'])
        return t.nn.BCELoss()(input=pred,
                              target=ground_truth)  # t.tensor([0, 0, 0, 0, 1.0, 0, 1.0], device=self.device))

    def get_loss_single_label(self, pred, y, label):
        ground_truth = [1.0, 0] if y['dx'].data.item() == label else [0, 1.0]
        ground_truth = t.tensor(ground_truth, device=self.device)
        return t.nn.BCELoss()(input=pred, target=ground_truth)

    def optimize_batch(self, batch_loss):
        [x.backward(retain_graph=True) for x in batch_loss]
        self.optim.step()
        self.optim.zero_grad()
        return

    def create_batch(self, data, batch_size=100):
        t.manual_seed(time.time())
        np.random.seed(int(time.time()))

        data_length = len(data[list(data.keys())[0]])
        batch_positions = t.randint(0, data_length, [batch_size], device=self.device)

        batch_data = {}
        mask = np.zeros(len(data['dx']))
        for p in batch_positions:
            mask[p] = 1
        for k in data.keys():
            if isinstance(data[k], list):
                batch_data[k] = list(compress(data[k], mask))
            else:
                batch_data[k] = t.masked_select(data[k], t.tensor(mask, dtype=t.uint8, device=self.device))
        batch_data = fragment_dict_data(batch_data)
        np.random.shuffle(batch_data)
        return batch_data

    def create_batch_for_label(self, data, label, batch_size=100):
        t.manual_seed(time.time())
        np.random.seed(int(time.time()))

        data = fragment_dict_data(data)
        np.random.shuffle(data)

        label_data = list(filter(lambda x: x['dx'] == label, data))
        nonlabel_data = list(filter(lambda x: x['dx'] != label, data))

        batch_data = []
        while len(batch_data) < batch_size:
            batch_data.append(label_data[np.random.randint(len(label_data))])
            batch_data.append(nonlabel_data[np.random.randint(len(nonlabel_data))])
        np.random.shuffle(batch_data)
        return batch_data

    def _tensorify(self, data):
        for k in data.keys():
            if not k.__contains__('id'):
                if k == 'age' or k == 'sex':
                    data[k] = t.tensor(data[k], device=self.device, dtype=t.float32)
                else:
                    data[k] = t.tensor(data[k], device=self.device, dtype=t.int64)
        return data

    @staticmethod
    def balance_classes(data, label):
        counts = Counter(data[label])
        max_cnt = np.max([counts[k] for k in counts])

        arr, keys = dict_to_array(data)
        label_pos = keys.index(label)

        labels_indices = {}

        for k in counts:
            labels_indices[k] = list(filter(lambda x: x[label_pos] == k, arr))

        new_data = []
        for k in labels_indices:
            if len(labels_indices[k]) == max_cnt:
                new_data += labels_indices[k]
            else:
                sampled_pos = np.random.randint(len(labels_indices[k]), size=max_cnt)
                [new_data.append(labels_indices[k][x]) for x in sampled_pos]
        new_data = arr_to_dict(new_data, keys)
        return new_data

    @staticmethod
    def _use_est(e, i, act=lambda x: x):
        o = act(e[0](i))
        for ee in e[1:]:
            o = act(ee(o))
        return o

    @staticmethod
    def _calculate_cnn_input_limit(cnn_sizes):
        lim = 0
        for i in range(len(cnn_sizes) - 1, -1, -1):
            cnn = cnn_sizes[i]
            lim = (lim - 1) * cnn['stride'] + cnn['kernel']
        return lim

    @staticmethod
    def _calculate_cnn_output_shape(input_shape, cnn_sizes):
        _h = lambda x, y: int((x - y['kernel']) / y['stride'] + 1)
        for c in cnn_sizes:
            input_shape[0] = _h(input_shape[0], c)
            input_shape[1] = _h(input_shape[1], c)
        return input_shape

    @staticmethod
    def multiclass_get_results(data, model_path):
        return
